# YouTube-8M Custom Code

Folder containing custom code for [YouTube-8M](https://github.com/google/youtube-8m) interop with the [AudioSet](https://research.google.com/audioset/) data set.

Core changes:

- the `context` of TFRecords in AudioSet uses `video_id` as opposed to YouTube8m which uses `id`; all instances of `id` have been changed to `video_id`
