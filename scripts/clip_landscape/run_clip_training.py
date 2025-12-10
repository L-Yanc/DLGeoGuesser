from src.dl_geoguesser.vision.clip_landscape.training import train_clip_head

if __name__ == "__main__":
    stats = train_clip_head("configs/clip_landscape.yaml")
    print(stats)
