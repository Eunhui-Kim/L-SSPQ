python ResIBShift34-dorefa-s1.py --wup_length 30 --step_size 30 --log './train_log/ResIBShift34-dorefa-clr-w30s30-s1-2'
python ResIBShift34-dorefa-s2.py --wup_length 30 --step_size 30 --max_epoch 10 --load './train_log/ResIBShift34-dorefa-clr-w30s30-s1-2/checkpoint' --log 'train_log/ResIBShift34-dorefa-clr-w30s30-s2-2'
