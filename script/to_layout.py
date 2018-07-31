import hack_path  # noqa
import time
import click 

from lsun_room.label import ColorLayout, color_palette

@click.command()
@click.option('--dataset_root', default='../data/lsun_room/')
def main(dataset_root):
    c = ColorLayout()
    b = c.to_layout(ColorLayout.frontal)
    print(b)
    
if __name__ == '__main__':
    main()
