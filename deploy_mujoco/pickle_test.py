import pickle
with open("/home/hpx/HPX_LOCO_2/whole_body_tracking/data_save.pkl", "rb") as f:
    data = pickle.load(f)
i = [d["time_step"] for d in data]
print(i)