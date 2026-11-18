import torch

from src.config import SplitConfig, TARGET_CLASS_TO_IDX, VERIFIED_TARGET_CLASSES
from src.data.dataset import MMActMultimodalDataset
from src.data.indexer import MMActIndexer, PairedSample
from src.data.splits import DatasetSplitter


def make_video(root, subject, scene, session, camera, action):
    path = root / "trimmed" / "video" / "cross_scene" / "trainval" / subject / camera / scene / session / f"{action}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_video_only_indexing_needs_no_sensor_files_and_filters_classes(tmp_path):
    root = tmp_path / "videos"
    valid = make_video(root, "subject01", "scene1", "session1", "cam1", "Waving Hand")
    make_video(root, "subject1", "scene1", "session1", "cam2", "jumping")
    unknown = make_video(root, "subject1", "scene1", "session1", "cam3", "flying")

    report = MMActIndexer(root).index_videos()
    assert report.total_video_files_found == 3
    assert report.paired_samples_count == 2
    assert {s.action for s in report.paired_samples} == {"waving_hand", "jumping"}
    assert report.unknown_label_files == [str(unknown)]
    sample = next(s for s in report.paired_samples if s.video_path == valid)
    assert sample.sample_id == "subject1_scene1_session1_cam1_waving_hand"
    assert sample.label_idx == TARGET_CLASS_TO_IDX["waving_hand"]
    assert sample.sensor_paths == {}


def test_video_only_subject_split_counts_ids_and_membership():
    samples = []
    # Synthetic fixture matches the verified 3,830 / 945 / 852 partition sizes.
    for subject_num, count in [(i, 273 + (1 if i <= 8 else 0)) for i in range(1, 15)]:
        for n in range(count):
            action = VERIFIED_TARGET_CLASSES[n % len(VERIFIED_TARGET_CLASSES)]
            samples.append(PairedSample(f"id_{subject_num}_{n}", action, TARGET_CLASS_TO_IDX[action],
                f"subject{subject_num}", f"scene{n % 4 + 1}", f"session{n + 1}", "cam1", Path(f"{subject_num}_{n}.mp4"), {}))
    for subject_num in range(15, 18):
        for n in range(315):
            action = VERIFIED_TARGET_CLASSES[n % len(VERIFIED_TARGET_CLASSES)]
            samples.append(PairedSample(f"id_{subject_num}_{n}", action, TARGET_CLASS_TO_IDX[action],
                f"subject{subject_num}", f"scene{n % 4 + 1}", f"session{n + 1}", "cam1", Path(f"{subject_num}_{n}.mp4"), {}))
    for subject_num in range(18, 21):
        for n in range(284):
            action = VERIFIED_TARGET_CLASSES[n % len(VERIFIED_TARGET_CLASSES)]
            samples.append(PairedSample(f"id_{subject_num}_{n}", action, TARGET_CLASS_TO_IDX[action],
                f"subject{subject_num}", f"scene{n % 4 + 1}", f"session{n + 1}", "cam1", Path(f"{subject_num}_{n}.mp4"), {}))

    train, val, test = DatasetSplitter(SplitConfig(split_strategy="subject")).split(samples)
    assert (len(train), len(val), len(test)) == (3830, 945, 852)
    assert {s.subject for s in train} == {f"subject{i}" for i in range(1, 15)}
    assert {s.subject for s in val} == {f"subject{i}" for i in range(15, 18)}
    assert {s.subject for s in test} == {f"subject{i}" for i in range(18, 21)}
    assert len({s.sample_id for s in test}) == 852


def test_video_only_dataset_does_not_read_sensors(tmp_path):
    class FakeVideoLoader:
        class Config:
            num_frames, frame_height, frame_width, channels = 16, 8, 8, 3
            sampling_strategy = "uniform"
        config = Config()

        def load_and_preprocess(self, _path):
            return torch.zeros(3, 16, 8, 8)

    class ForbiddenSensorLoader:
        class Config:
            modalities, window_size, primary_rate_hz = ["acc_phone", "acc_watch", "gyro"], 100, 100
        config = Config()

        def load_and_preprocess(self, _paths):
            raise AssertionError("Video-only dataset attempted to read sensor files")

    sample = PairedSample("video_sample", "walking", TARGET_CLASS_TO_IDX["walking"], "subject1",
        "scene1", "session1", "cam1", tmp_path / "no_video_needed.mp4", {})
    dataset = MMActMultimodalDataset([sample], video_loader=FakeVideoLoader(), sensor_loader=ForbiddenSensorLoader(),
                                     enable_cache=False, mode="video_only")
    video, sensor, label, sample_id = dataset[0]
    assert video.shape == (3, 16, 8, 8)
    assert sensor.shape == (9, 100)
    assert label == TARGET_CLASS_TO_IDX["walking"]
    assert sample_id == "video_sample"


def test_indexer_none_uses_three_default_sensor_requirements(tmp_path):
    assert MMActIndexer(tmp_path, required_sensor_modalities=None).required_sensors == [
        "acc_phone", "acc_watch", "gyro"
    ]


def test_indexer_empty_list_means_video_only_and_skips_sensor_discovery(tmp_path, monkeypatch):
    root = tmp_path / "video_only"
    video = make_video(root, "subject01", "scene1", "session1", "cam1", "walking")
    indexer = MMActIndexer(root, required_sensor_modalities=[])
    assert indexer.required_sensors == []

    def forbidden_sensor_discovery(*_args, **_kwargs):
        raise AssertionError("Video-only indexing must not search for sensor CSVs")

    monkeypatch.setattr(indexer, "parse_sensor_path", forbidden_sensor_discovery)
    monkeypatch.setattr(indexer, "find_files_recursively", lambda path, extensions: (
        [video] if extensions == {".mp4"} else forbidden_sensor_discovery()
    ))
    report = indexer.index()
    assert report.paired_samples_count == 1
    assert report.paired_samples[0].sample_id == "subject1_scene1_session1_cam1_walking"
    assert report.paired_samples[0].sensor_paths == {}


def test_default_multimodal_indexer_still_excludes_video_missing_required_sensors(tmp_path):
    root = tmp_path / "multimodal"
    video = make_video(root, "subject1", "scene1", "session1", "cam1", "walking")

    indexer = MMActIndexer(root)
    assert indexer.required_sensors == ["acc_phone", "acc_watch", "gyro"]
    report = indexer.index()
    assert report.paired_samples_count == 0
    assert len(report.missing_sensors) == 1
    assert report.missing_sensors[0]["video_path"] == str(video)


def test_multimodal_indexer_pairs_when_all_required_sensors_exist(tmp_path):
    root = tmp_path / "paired"
    video = make_video(root, "subject1", "scene1", "session1", "cam1", "walking")
    for modality in ("acc_phone_clip", "acc_watch_clip", "gyro_clip"):
        sensor = root / "trimmed" / "sensors" / modality / "subject1" / "scene1" / "session1" / "walking.csv"
        sensor.parent.mkdir(parents=True, exist_ok=True)
        sensor.write_text("time,x,y,z\n0,0,0,0\n")

    report = MMActIndexer(root).index()
    assert report.paired_samples_count == 1
    assert report.paired_samples[0].video_path == video
    assert set(report.paired_samples[0].sensor_paths) == {"acc_phone", "acc_watch", "gyro"}
