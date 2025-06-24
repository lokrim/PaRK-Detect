import scipy.io
mat = scipy.io.loadmat('link_key_points_final/104_mask.mat')
print(mat.keys())  # Should show: ['if_key_points', 'all_key_points_position', 'anchor_link']
print(mat['if_key_points'].shape)        # Should be (1, 64, 64)
print(mat['all_key_points_position'].shape)  # Should be (2, 64, 64)
print(mat['anchor_link'].shape)          # Should be (8, 64, 64)