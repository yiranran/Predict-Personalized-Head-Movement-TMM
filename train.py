# source /home4/yiran/venv/tensorflown/bin/activate
import os
import argparse


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=18, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)

    return parser.parse_args()


def execute_os_cmd(cmd, logging=True):
    if logging:
        print('\n\x1b[1;37;44m' + '+ ' + cmd + '\x1b[0m\n')
    assert os.system(cmd) == 0


if __name__ == '__main__':
    opt = opts()
    gpu_id = opt.gpu_id
    uid = opt.id

    # put a fps=25 video into Data/ folder
    if not os.path.exists('render-to-video/checkpoints/seq_p2p/{}_13-2/60_net_G.pth'.format(uid)):
        command_preprocess = 'cd Data/; python extract_frame2.py {}.mp4 0;cd ../WM3DR; python demo_video_fiona.py --gpu_id {} --video_path ../Data/{}'.format(
            uid, gpu_id, uid)
        print('preprocessing ...')
        os.system(command_preprocess)

        command_train = 'cd render-to-video/; python train_19news_nomem.py {} {}'.format(uid, gpu_id)
        print('training ...')
        execute_os_cmd(command_train)