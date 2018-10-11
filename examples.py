from models.unet import build_model

m = build_model((3, 3), [64, 128, 256, 512], dropout=0.1, padding='same')