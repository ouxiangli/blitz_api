import numpy as np

def PreNormalize2D(results):
    # print("************************PreNormalize2D********************")
    h, w = results['img_shape']
    results['keypoint'][..., 0] = (results['keypoint'][..., 0] - (w / 2)) / (w / 2)
    results['keypoint'][..., 1] = (results['keypoint'][..., 1] - (h / 2)) / (h / 2)
    return results

def GenSkeFeat(results):
    # print("********************GenSkeFeat*****************")
    keypoint = results.pop('keypoint')
    keypoint_score = results.pop('keypoint_score')
    results['keypoint'] = np.concatenate([keypoint, keypoint_score[..., None]], -1)
    return results

def _get_train_clips(num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        old_num_frames = num_frames
        pi = [1, 1]
        ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
        num_frames = int(ratio * num_frames)
        off = np.random.randint(old_num_frames - num_frames + 1)

        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset

        inds = inds + off
        num_frames = old_num_frames

        allinds.append(inds)

        return np.concatenate(allinds)


def UniformSampleFrames(results, clip_len=100):
            # print("******************UniformSample*********************")
    num_frames = results['total_frames']
    
    inds = _get_train_clips(num_frames, clip_len)

    inds = np.mod(inds, num_frames)
    start_index = 0
    inds = inds + start_index

    if 'keypoint' in results:
        kp = results['keypoint']
        assert num_frames == kp.shape[1]
        num_person = kp.shape[0]
        num_persons = [num_person] * num_frames
        for i in range(num_frames):
            j = num_person - 1
            while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                j -= 1
            num_persons[i] = j + 1
        transitional = [False] * num_frames
        for i in range(1, num_frames - 1):
            if num_persons[i] != num_persons[i - 1]:
                transitional[i] = transitional[i - 1] = True
            if num_persons[i] != num_persons[i + 1]:
                transitional[i] = transitional[i + 1] = True
        inds_int = inds.astype(np.int)
        coeff = np.array([transitional[i] for i in inds_int])
        inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

    results['frame_inds'] = inds.astype(np.int)
    results['clip_len'] = clip_len
    results['frame_interval'] = None
    results['num_clips'] = 1
    return results

def _load_kp(kp, frame_inds):
    return kp[:, frame_inds].astype(np.float32)
    
def PoseDecode(results):
    # print("*****************PoseDecode********************")

    offset = results.get('offset', 0)
    frame_inds = results['frame_inds'] + offset

    if 'keypoint' in results:
        results['keypoint'] = _load_kp(results['keypoint'], frame_inds)

    return results


def FormatGCNInput(results, mode = 'zero', num_person = 1):
    # print("********************FormatGCNInput******************")
    """Performs the FormatShape formatting.

    Args:
        results (dict): The resulting dict to be modified and passed
            to the next transform in pipeline.
    """
    keypoint = results['keypoint']
    if 'keypoint_score' in results:
        keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

    # M T V C
    if keypoint.shape[0] < num_person:
        pad_dim = num_person - keypoint.shape[0]
        pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
        keypoint = np.concatenate((keypoint, pad), axis=0)
        if mode == 'loop' and keypoint.shape[0] == 1:
            for i in range(1, num_person):
                keypoint[i] = keypoint[0]

    elif keypoint.shape[0] > num_person:
        keypoint = keypoint[:num_person]

    M, T, V, C = keypoint.shape
    nc = results.get('num_clips', 1)
    assert T % nc == 0
    keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
    results['keypoint'] = np.ascontiguousarray(keypoint)
    return results


def Collect(results):
    data = {}
    for key in ['keypoint', 'label']:
        data[key] = results[key]
    return data