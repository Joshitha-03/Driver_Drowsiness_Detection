from feature_extraction import extract_features_from_images

extract_features_from_images("train_data/notdrowsy","features/alert.csv",0)
extract_features_from_images("train_data/drowsy","features/drowsy.csv",1)
