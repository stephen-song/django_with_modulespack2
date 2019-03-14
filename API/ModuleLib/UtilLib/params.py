

logdir_path = './logdir'
case = 'out'


class Default:
    # signal processing
    sr = 16000                          # sample rate  16000hz 16000samples/s
    frame_shift = 0.006                 # Window sliding distance(percentage)
    frame_length = 0.071   # window size(percentage 7.1%)
    hop_length = 96                     # 80 samples.  This is dependent on the frame_shift. # window size(samples)  = sr * frame_length(recommend value is 160，which means 10ms)
    win_length = 1136                   # 400 samples. This is dependent on the frame_length.# Window sliding distance(samples) = sr * frame_shift (common value is win_length // 4)
    n_fft = 1136                        # number of FFT components(filters) OR FFT window size(samples) Generally slightly larger than win_length，It does the same thing，It's used for framing, or you can think of it as taking a discrete Fourier transform of every sample point, so it's about the same size as win_length
    preemphasis = 0.97                  # Coefficient of pre-emphasis[Emphasize high frequency range of the waveform by increasing power(squared amplitude).]
    n_mels = 90                         # number of Mel bands to generate (which is same to feature_bin_count)
    n_mfcc = 60                         # number of MFCC output components (DCT filters)
    n_iter = 60                         # Number of inversion iterations
    duration = 2                        # Length of each audio clip to be analyzed duration(s)
    max_db = 40
    min_db = -50

    # model
    hidden_units = 256                  # alias = E
    num_banks = 16
    num_highway_blocks = 4
    norm_type = 'ins'                   # a normalizer function. value = bn, ln, ins, or None
    t = 1.0                             # temperature
    dropout_rate = 0.2

    # train
    batch_size = 32


class Train1:
    # path
    data_path = 'D:/pycharm_proj/corpus/data/lisa/data/timit/raw/TIMIT/TRAIN/*/*/*.WAV'

    # model
    hidden_units = 128                  # alias = E
    num_banks = 8
    num_highway_blocks = 4
    norm_type = 'ins'                   # a normalizer function. value = bn, ln, ins, or None
    t = 1.0                             # temperature
    dropout_rate = 0.2

    # train
    batch_size = 32#20
    lr = 0.0003
    num_epochs = 1000
    steps_per_epoch = 100
    save_per_epoch = 2
    num_gpu = 1


class Train2:
    # path
    data_path = 'net2_data/train/cmu_us_bdl_arctic/wav/*.wav'

    # model
    hidden_units = 256                  # alias = E
    num_banks = 8
    num_highway_blocks = 8
    norm_type = 'ins'                   # a normalizer function. value = bn, ln, ins, or None
    t = 1.0                             # temperature
    dropout_rate = 0.2

    # train
    batch_size = 32#50                     # 50
    lr = 0.0003
    lr_cyclic_margin = 0.
    lr_cyclic_steps = 5000
    clip_value_max = 3.                 # max value of cliped grad
    clip_value_min = -3.                # min value of cliped grad
    clip_norm = 10                      # max value of global norm # The global norm is the sum of norm for **all** gradients.
    num_epochs = 1000                   # 10000
    steps_per_epoch = 100
    save_per_epoch = 50
    test_per_epoch = 1
    num_gpu = 1


class Test1:
    # path
    data_path = 'D:/pycharm_proj/corpus/data/lisa/data/timit/raw/TIMIT/TEST/*/*/*.WAV'

    # test
    batch_size = 32


class Test2:
    data_path = 'net2_data/test/*.wav'
    # test
    batch_size = 3


class Convert:
    # pathD:\deepvoice\convertaudio
    data_path = 'net2_data/test/*.wav'

    # convert
    one_full_wav = True
    batch_size = 1
    emphasis_magnitude = 1.6   #1.2