python demo.py \
--save_path <save_path> \
--setting dyml \
--num_workers 0 \
--device_type DDP --is_distributed --world_size 2 \
--save_name dyml-vehicle \
--batch_size 20 --test_batch_size 40 \
--lr_trunk 0.00001 \
--lr_embedder 0.0001 \
--lr_collector 0.01 \
--reduction_size 512 256 128 64 \
