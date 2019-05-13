
# 1 -----------------------------------------
mod_training_file = './data/mod_train0.p'
mod_testing_file = './data/mod_test0.p'

X_train, y_train = train['features'], train['labels']
for idx in range(len(X_train)):
    X_train[idx, :, :, :] = cv2.cvtColor(X_train[idx, :, :, :], cv2.COLOR_RGB2YUV)

with open(mod_training_file, mode='wb') as f:
    pickle.dump(X_train, f)

X_test, y_test = test['features'], test['labels']
for idx in range(len(X_test)):
    X_test[idx, :, :, :] = cv2.cvtColor(X_test[idx, :, :, :], cv2.COLOR_RGB2YUV)

with open(mod_testing_file, mode='wb') as f:
    pickle.dump(X_test, f)


# 2 -----------------------------------------
mod_training_file = './data/mod_train0.p'
mod_testing_file = './data/mod_test0.p'

def rotate_image(image, max_angle =15):
    rotate_out = rotate(image, np.random.uniform(-max_angle, max_angle), mode='edge')
    return rotate_out

def translate_image(image, max_trans = 5, height=32, width=32):
    translate_x = max_trans*np.random.uniform() - max_trans/2
    translate_y = max_trans*np.random.uniform() - max_trans/2
    translation_mat = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    trans = cv2.warpAffine(image, translation_mat, (height, width))
    return trans


def projection_transform(image, max_warp=0.8, height=32, width=32):
    # Warp Location
    d = height * 0.3 * np.random.uniform(0, max_warp)

    # Warp co-ordinates
    tl_top = np.random.uniform(-d, d)  # Top left corner, top margin
    tl_left = np.random.uniform(-d, d)  # Top left corner, left margin
    bl_bottom = np.random.uniform(-d, d)  # Bottom left corner, bottom margin
    bl_left = np.random.uniform(-d, d)  # Bottom left corner, left margin
    tr_top = np.random.uniform(-d, d)  # Top right corner, top margin
    tr_right = np.random.uniform(-d, d)  # Top right corner, right margin
    br_bottom = np.random.uniform(-d, d)  # Bottom right corner, bottom margin
    br_right = np.random.uniform(-d, d)  # Bottom right corner, right margin

    ##Apply Projection
    transform = ProjectiveTransform()
    transform.estimate(np.array((
        (tl_left, tl_top),
        (bl_left, height - bl_bottom),
        (height - br_right, height - br_bottom),
        (height - tr_right, tr_top)
    )), np.array((
        (0, 0),
        (0, height),
        (height, height),
        (height, 0)
    )))
    output_image = warp(image, transform, output_shape=(height, width), order=1, mode='edge')
    return output_image

X_train, y_train = train['features'], train['labels']

num_rot = 5
y_train1 = np.matlib.repmat(y_train, num_rot, 1)
y_train1 = y_train1.T.reshape(-1)
X_train1 = np.zeros([len(X_train)*num_rot, 32, 32, 3])
for idx in range(len(X_train)):
    for idx1 in range(num_rot):
        k = idx * num_rot + idx1
        X_train1[k, :, :, :] = rotate_image(X_train[idx, :, :, :], max_angle=15)

with open(mod_training_file, mode='wb') as f:
    pickle.dump(X_train1, f)

# 3 -----------------------------------------
num_rot = 1
y_train2 = np.matlib.repmat(y_train1, num_rot, 1)
y_train2 = y_train2.T.reshape(-1)
X_train2 = np.zeros([len(X_train1)*num_rot, 32, 32, 3])
for idx in range(len(X_train1)):
    for idx1 in range(num_rot):
        k = idx * num_rot + idx1
        X_train2[k, :, :, :] = projection_transform(X_train1[idx, :, :, :])

with open(mod_training_file, mode='wb') as f:
    pickle.dump(X_train2, f)


# 4 --------------------------------------------
X_train, y_train = train['features'], train['labels']
for idx in range(len(X_train)):
    X_train[idx, :, :, 0] = cv2.equalizeHist(X_train[idx, :, :, 0])
    X_train[idx, :, :, 1] = cv2.equalizeHist(X_train[idx, :, :, 1])
    X_train[idx, :, :, 2] = cv2.equalizeHist(X_train[idx, :, :, 2])

# with open(mod_training_file, mode='wb') as f:
#     pickle.dump(X_train, f)

X_test, y_test = test['features'], test['labels']
for idx in range(len(X_test)):
    X_test[idx, :, :, 0] = cv2.equalizeHist(X_test[idx, :, :, 0])
    X_test[idx, :, :, 1] = cv2.equalizeHist(X_test[idx, :, :, 1])
    X_test[idx, :, :, 2] = cv2.equalizeHist(X_test[idx, :, :, 2])


for idx in range(27):
    plt.figure(idx)
    image = X_train2[idx].squeeze()
    plt.imshow(image)