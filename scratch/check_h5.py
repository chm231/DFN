import h5py

with h5py.File("storage/data/dfn_export_for_python.h5", "r") as f:
    def print_item(name, obj):
        print(name)
        if isinstance(obj, h5py.Dataset):
            print("  Dataset:", obj.shape, obj.dtype)
            if obj.size < 50:
                print("  Value:", obj[:])
    f.visititems(print_item)
