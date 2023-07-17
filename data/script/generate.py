import os
import argparse
import logging


def write_txt(file, content):
    with open(file, "w") as f:
        for item in content:
            f.write(str(item) + "\n")
    return None


def processed(args):
    logging.getLogger().setLevel(logging.INFO)
    # logging.basicConfig(
    #     filename='log.txt',
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
    #     level=logging.INFO,
    # )
    save_file = args.save_file
    data_dir = args.data_dir
    logging.info('Scaning data dir {}'.format(data_dir))
    data = os.listdir(data_dir)
    write_txt(save_file, [os.path.join(data_dir, item) for item in data])
    logging.info('Sucessfully generate the path list file')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the coco data (dir)')
    parser.add_argument('--save_file', type=str, required=True, help='Path to saving the sample list (.txt)')

    args = parser.parse_args()
    processed(args)
