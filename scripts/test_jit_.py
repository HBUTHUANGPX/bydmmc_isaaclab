import torch
frozen_actor_path = "/home/hpx/HPX_LOCO_2/whole_body_tracking/logs/rsl_rl/temp/exported/policy.pt"
frozen_actor = torch.jit.load(frozen_actor_path)
print(frozen_actor)
print(frozen_actor.actor)
# print(frozen_actor.normalizer)
# print(frozen_actor.normalizer.training)
for layer in frozen_actor.actor.children():
    if isinstance(layer, torch.nn.Linear):
        print(layer.original_name)
    else:
        print("0",layer.original_name)
        print(type(layer))
