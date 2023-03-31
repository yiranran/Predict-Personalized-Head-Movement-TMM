# source /home4/yiran/venv/tensorflown/bin/activate
import os
import argparse


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=18, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('-a', '--audio', type=str)
    return parser.parse_args()


def execute_os_cmd(cmd, logging=True):
    if logging:
        print('\n\x1b[1;37;44m' + '+ ' + cmd + '\x1b[0m\n')
    assert os.system(cmd) == 0


if __name__ == '__main__':
    opt = opts()
    gpu_id = opt.gpu_id
    uid = opt.id

    if not os.path.exists('render-to-video/checkpoints/seq_p2p/{}_13-2/60_net_G.pth'.format(uid)):
        print("ERROR: Not trained yet!")
        exit(-1)

    audio_name = os.path.basename(opt.audio)
    if not os.path.exists(os.path.join("Audio/audio", audio_name)):
        cmd_link = f"ln -s {opt.audio} Audio/audio/"
        execute_os_cmd(cmd_link)

    audio_id = os.path.splitext(audio_name)[0]
    cmd_test = 'cd Audio/code/; python test_personalized_rev3_nomem.py --audiobasen {} --person {} --gpu_id {}'.format(
        audio_id, uid, gpu_id)
    execute_os_cmd(cmd_test)
