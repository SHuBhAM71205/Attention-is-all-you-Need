import mmap
import array
# bo is byte offset
byte_offset_loc_en = "./en_offset.bo"
byte_offset_loc_hi = "./hi_offset.bo"

data_dir = "../Data"

def save_offset(offset_lst,file_loc):
    arr = array.array('Q',offset_lst)
    with open(file_loc,"wb") as f:
        arr.tofile(f)

en_offsets = []
hi_offsets = []

with open(f"{data_dir}/parallel-n/en-hi.en", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    en_offsets.append(0)
    size = mm.size()
    for i in range(size):
        if mm[i] == 10:
            en_offsets.append(i + 1)
    mm.close()

with open(f"{data_dir}/parallel-n/en-hi.hi", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    hi_offsets.append(0)
    size = mm.size()
    for i in range(size):
        if mm[i] == 10:
            hi_offsets.append(i + 1)
    mm.close()

if len(en_offsets) == len(hi_offsets):
    save_offset(en_offsets,byte_offset_loc_en)
    save_offset(hi_offsets,byte_offset_loc_hi)

else:
    print("the en_hindi shoud have same length")