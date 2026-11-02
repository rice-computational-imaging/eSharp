set e
set x

python "script/dataset_preprocess/MVSEC/generate_mvsec_dataset.py" \
--base_dir="/path/to/data/event3d/MVSEC/outdoor_day1" \
--event_dir="outdoor_day1_data" \
--hdf5 \
--time_encoding="VAE_ROBUST" \
--save_dir="encoded_full" \
# --vmin=0 --vmax=0.06 --interval_vmax=0.06 \

