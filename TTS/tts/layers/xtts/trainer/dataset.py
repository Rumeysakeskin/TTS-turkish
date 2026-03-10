import os
import random
import sys
from collections import Counter

import torch
import torch.nn.functional as F
import torch.utils.data

from TTS.tts.models.xtts import load_audio

torch.set_num_threads(1)


def key_samples_by_col(samples, col):
    """Returns a dictionary of samples keyed by language."""
    samples_by_col = {}
    for sample in samples:
        col_val = sample[col]
        assert isinstance(col_val, str)
        if col_val not in samples_by_col:
            samples_by_col[col_val] = []
        samples_by_col[col_val].append(sample)
    return samples_by_col


def get_prompt_slice(gt_path, max_sample_length, min_sample_length, sample_rate, is_eval=False):
    rel_clip = load_audio(gt_path, sample_rate)
    # if eval uses a middle size sample when it is possible to be more reproducible
    if is_eval:
        sample_length = int((min_sample_length + max_sample_length) / 2)
    else:
        sample_length = random.randint(min_sample_length, max_sample_length)
    gap = rel_clip.shape[-1] - sample_length
    if gap < 0:
        sample_length = rel_clip.shape[-1] // 2
    gap = rel_clip.shape[-1] - sample_length

    # if eval start always from the position 0 to be more reproducible
    if is_eval:
        rand_start = 0
    else:
        rand_start = random.randint(0, gap)

    rand_end = rand_start + sample_length
    rel_clip = rel_clip[:, rand_start:rand_end]
    rel_clip = F.pad(rel_clip, pad=(0, max_sample_length - rel_clip.shape[-1]))
    cond_idxs = [rand_start, rand_end]
    return rel_clip, rel_clip.shape[-1], cond_idxs


class XTTSDataset(torch.utils.data.Dataset):
    def __init__(self, config, samples, tokenizer, sample_rate, is_eval=False):
        self.config = config
        model_args = config.model_args
        self.failed_samples = set()
        self.debug_failures = model_args.debug_loading_failures
        self.max_conditioning_length = model_args.max_conditioning_length
        self.min_conditioning_length = model_args.min_conditioning_length
        self.is_eval = is_eval
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_wav_len = model_args.max_wav_length
        self.max_text_len = model_args.max_text_length
        self.use_masking_gt_prompt_approach = model_args.gpt_use_masking_gt_prompt_approach
        assert self.max_wav_len is not None and self.max_text_len is not None

        excluded_emotions = {e.lower() for e in getattr(model_args, "emotion_exclude", [])}
        allowed_emotions = getattr(model_args, "emotion_labels", None)
        if allowed_emotions:
            allowed_emotions = [e.lower() for e in allowed_emotions]

        processed_samples = []
        for sample in samples:
            emotion = sample.get("emotion")
            emotion = str(emotion).lower() if emotion is not None else "neutral"
            if allowed_emotions and emotion not in allowed_emotions:
                continue
            if emotion in excluded_emotions:
                continue
            normalized_sample = sample.copy()
            normalized_sample["emotion"] = emotion
            processed_samples.append(normalized_sample)

        if not processed_samples:
            raise RuntimeError("No samples left after applying emotion filtering.")

        self.samples_list = processed_samples
        self.global_emotion_counts = Counter(sample["emotion"] for sample in self.samples_list)

        # determine emotion set and mapping
        if allowed_emotions:
            present_emotions = [e for e in allowed_emotions if self.global_emotion_counts.get(e, 0) > 0]
        else:
            present_emotions = sorted(self.global_emotion_counts.keys())
            if "neutral" in present_emotions:
                present_emotions = ["neutral"] + [e for e in present_emotions if e != "neutral"]

        if not present_emotions:
            raise RuntimeError("Unable to determine emotion labels for dataset.")

        self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(present_emotions)}
        self.id_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_id.items()}

        # compute global class weights (inverse frequency)
        total_samples = sum(self.global_emotion_counts[e] for e in present_emotions)
        num_classes = len(present_emotions)
        self.class_weights = {}
        for emotion in present_emotions:
            count = self.global_emotion_counts.get(emotion, 0)
            if count == 0:
                continue
            self.class_weights[emotion] = total_samples / (num_classes * count)

        print(" > Emotion distribution:", dict(self.global_emotion_counts))
        if self.class_weights:
            print(" > Emotion class weights:", {k: round(v, 4) for k, v in self.class_weights.items()})

        if not is_eval:
            random.seed(config.training_seed)
            random.shuffle(self.samples_list)
            # organize by language and emotion for balanced sampling
            self.samples_by_language = key_samples_by_col(self.samples_list, "language")
            self.samples_by_language_emotion = {}
            self.language_emotion_weights = {}
            for language, lang_samples in self.samples_by_language.items():
                emotion_groups = key_samples_by_col(lang_samples, "emotion")
                # ensure groups align with known emotions
                emotion_groups = {
                    emotion: items for emotion, items in emotion_groups.items() if emotion in present_emotions
                }
                if not emotion_groups:
                    continue
                self.samples_by_language_emotion[language] = emotion_groups
                lang_counts = {emotion: len(items) for emotion, items in emotion_groups.items()}
                lang_total = sum(lang_counts.values())
                lang_classes = len(lang_counts)
                weights = []
                emotions = []
                for emotion, count in lang_counts.items():
                    weight = lang_total / (lang_classes * count)
                    weights.append(weight)
                    emotions.append(emotion)
                weight_sum = sum(weights)
                probs = [w / weight_sum for w in weights]
                self.language_emotion_weights[language] = (emotions, probs)
            print(" > Sampling by language:", self.samples_by_language_emotion.keys())
        else:
            self.samples = self.samples_list
            # for evaluation load and check samples that are corrupted to ensures the reproducibility
            self.check_eval_samples()

    def check_eval_samples(self):
        print(" > Filtering invalid eval samples!!")
        new_samples = []
        for sample in self.samples:
            try:
                tseq, _, wav, _, _, _ = self.load_item(sample)
            except:
                continue
            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            if (
                wav is None
                or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len)
                or (self.max_text_len is not None and tseq.shape[0] > self.max_text_len)
            ):
                continue
            new_samples.append(sample)
        self.samples = new_samples
        print(" > Total eval samples after filtering:", len(self.samples))

    def get_text(self, text, lang):
        tokens = self.tokenizer.encode(text, lang)
        tokens = torch.IntTensor(tokens)
        assert not torch.any(tokens == 1), f"UNK token found in {text} -> {self.tokenizer.decode(tokens)}"
        # The stop token should always be sacred.
        assert not torch.any(tokens == 0), f"Stop token found in {text}"
        return tokens

    def load_item(self, sample):
        text = str(sample["text"])
        tseq = self.get_text(text, sample["language"])
        audiopath = sample["audio_file"]
        wav = load_audio(audiopath, self.sample_rate)
        if text is None or len(text.strip()) == 0:
            raise ValueError
        if wav is None or wav.shape[-1] < (0.5 * self.sample_rate):
            # Ultra short clips are also useless (and can cause problems within some models).
            raise ValueError

        if self.use_masking_gt_prompt_approach:
            # get a slice from GT to condition the model
            cond, _, cond_idxs = get_prompt_slice(
                audiopath, self.max_conditioning_length, self.min_conditioning_length, self.sample_rate, self.is_eval
            )
            # if use masking do not use cond_len
            cond_len = torch.nan
        else:
            ref_sample = (
                sample["reference_path"]
                if "reference_path" in sample and sample["reference_path"] is not None
                else audiopath
            )
            cond, cond_len, _ = get_prompt_slice(
                ref_sample, self.max_conditioning_length, self.min_conditioning_length, self.sample_rate, self.is_eval
            )
            # if do not use masking use cond_len
            cond_idxs = torch.nan

        return tseq, audiopath, wav, cond, cond_len, cond_idxs

    def __getitem__(self, index):
        if self.is_eval:
            sample = self.samples[index]
            sample_id = str(index)
        else:
            # select a random language
            language_choices = list(self.samples_by_language_emotion.keys())
            lang = random.choice(language_choices)
            # select emotion based on balancing weights
            emotions, probs = self.language_emotion_weights[lang]
            emotion = random.choices(emotions, weights=probs, k=1)[0]
            sample_list = self.samples_by_language_emotion[lang][emotion]
            index = random.randint(0, len(sample_list) - 1)
            sample = sample_list[index]
            # a unique id for each sample to deal with fails
            sample_id = f"{lang}_{emotion}_{index}"

        # ignore samples that we already know that is not valid ones
        if sample_id in self.failed_samples:
            if self.debug_failures:
                print(f"Ignoring sample {sample['audio_file']} because it was already ignored before !!")
            # call get item again to get other sample
            return self[1]

        # try to load the sample, if fails added it to the failed samples list
        try:
            tseq, audiopath, wav, cond, cond_len, cond_idxs = self.load_item(sample)
        except:
            if self.debug_failures:
                print(f"error loading {sample['audio_file']} {sys.exc_info()}")
            self.failed_samples.add(sample_id)
            return self[1]

        # check if the audio and text size limits and if it out of the limits, added it failed_samples
        if (
            wav is None
            or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len)
            or (self.max_text_len is not None and tseq.shape[0] > self.max_text_len)
        ):
            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            # It's hard to handle this situation properly. Best bet is to return the a random valid token and skew the dataset somewhat as a result.
            if self.debug_failures and wav is not None and tseq is not None:
                print(
                    f"error loading {sample['audio_file']}: ranges are out of bounds; {wav.shape[-1]}, {tseq.shape[0]}"
                )
            self.failed_samples.add(sample_id)
            return self[1]

        emotion = sample["emotion"]
        emotion_id = self.emotion_to_id[emotion]
        emotion_weight = self.class_weights.get(emotion, 1.0)

        res = {
            # 'real_text': text,
            "text": tseq,
            "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
            "filenames": audiopath,
            "conditioning": cond.unsqueeze(1),
            "cond_lens": torch.tensor(cond_len, dtype=torch.long)
            if cond_len is not torch.nan
            else torch.tensor([cond_len]),
            "cond_idxs": torch.tensor(cond_idxs) if cond_idxs is not torch.nan else torch.tensor([cond_idxs]),
            "emotion_id": torch.tensor(emotion_id, dtype=torch.long),
            "emotion_weight": torch.tensor(emotion_weight, dtype=torch.float32),
        }
        return res

    def __len__(self):
        if self.is_eval:
            return len(self.samples)
        return len(self.samples_list)

    def collate_fn(self, batch):
        # convert list of dicts to dict of lists
        B = len(batch)

        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        # stack for features that already have the same shape
        batch["wav_lengths"] = torch.stack(batch["wav_lengths"])
        batch["text_lengths"] = torch.stack(batch["text_lengths"])
        batch["conditioning"] = torch.stack(batch["conditioning"])
        batch["cond_lens"] = torch.stack(batch["cond_lens"])
        batch["cond_idxs"] = torch.stack(batch["cond_idxs"])
        batch["emotion_id"] = torch.stack(batch["emotion_id"])
        batch["emotion_weight"] = torch.stack(batch["emotion_weight"])

        if torch.any(batch["cond_idxs"].isnan()):
            batch["cond_idxs"] = None

        if torch.any(batch["cond_lens"].isnan()):
            batch["cond_lens"] = None

        max_text_len = batch["text_lengths"].max()
        max_wav_len = batch["wav_lengths"].max()

        # create padding tensors
        text_padded = torch.IntTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, max_wav_len)

        # initialize tensors for zero padding
        text_padded = text_padded.zero_()
        wav_padded = wav_padded.zero_()
        for i in range(B):
            text = batch["text"][i]
            text_padded[i, : batch["text_lengths"][i]] = torch.IntTensor(text)
            wav = batch["wav"][i]
            wav_padded[i, :, : batch["wav_lengths"][i]] = torch.FloatTensor(wav)

        batch["wav"] = wav_padded
        batch["padded_text"] = text_padded
        return batch
