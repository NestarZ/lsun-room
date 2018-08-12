import hack_path  # noqa
import time

import click

from lsun_room.item import DataItems
from lsun_room.edge import mapping_func


@click.command()
@click.option('--dataset_root', default='../data/lsun_room/')
def main(dataset_root):

    for phase in ['train', 'val']:
        print('==> iter for data in %s phase' % phase)
        s = time.time()
        dataset = DataItems(root=dataset_root, phase=phase)

        for i, e in enumerate(dataset.items):
            if e.type == 10:
                func = mapping_func(e.type)
                out = func(e, image_size=(404, 404), width=2)

                import cv2
                orgfn = "/app/data/lsun_room/images/" + e.name + ".jpg"
                org = cv2.imread(orgfn)
                cv2.imwrite('/app/out/org%s.png'%i, org)
                cv2.imwrite('/app/out/edge%s.png'%i, out)
                if i > 100:
                    break
        print('==> Done in %.4f sec.' % (time.time() - s))


if __name__ == '__main__':
    main()
