fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(test_wt[10,:,:,:], cmap='gray')
output_filename = 'test_wt_slice_10.png'
output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
plt.savefig(output_path)
plt.close()

y_pred_thresholded = test_wt > 0.4
fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(y_pred_thresholded[10,:,:,:], cmap='gray')
output_filename = 'test_wt_thresholded_slice_10.png'
output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
plt.savefig(output_path)
plt.close()

# Existing function definitions remain unchanged

for batch_x, batch_y in test_generator:
    mask_image = np.expand_dims(batch_y, axis=-1)
    y_predwts = model.predict(batch_x)
    y_pred = np.where(y_predwts < 0.2, 0.0, y_predwts).astype(np.float32)
    y_pred_thresholded = y_pred
    for i in range(len(batch_x)):
        dice = dice_coeff(batch_y[i], y_pred_thresholded[i])
        iou_value = iou(batch_y[i], y_pred_thresholded[i])
        dice_values.append(dice)
        iou_values.append(iou_value)
    if len(loss_values) >= len(test_generator):
        break

average_dice = np.mean(dice_values)
average_iou = np.mean(iou_values)

print("Average test dice: ", average_dice)
print("Average test IoU: ", average_iou)

example_case = 19

dwi_path = os.path.join(base_path, 'rawdata', 'sub-strokecase{}'.format("%04d" %example_case), 'ses-0001', 'dwi/', 'sub-strokecase{}_ses-0001_dwi.nii.gz'.format("%04d" % example_case))
mask_path = os.path.join(base_path, 'derivatives', 'sub-strokecase{}'.format("%04d" %example_case), 'ses-0001', 'sub-strokecase{}_ses-0001_msk.nii.gz'.format("%04d" % example_case))

dwi_image = nib.load(dwi_path).get_fdata()
mask_image = nib.load(mask_path).get_fdata()

img_resize = lambda img, dims: cv2.resize(img[:,:], dims)

dwi_image=img_resize(dwi_image, (112, 112))
mask_image=img_resize(mask_image, (112, 112))
print("dwi_image.shape: ", dwi_image.shape)
print("mask_image.shape: ", mask_image.shape)

fig, (ax1, ax2) = plt.subplots(1, 2)

slice2show=31
ax1.imshow(dwi_image[:,:,slice2show], cmap='gray')
ax1.set_title('Dwi')
ax1.set_axis_off()
output_filename = 'dwi_image_slice_31.png'
output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
plt.savefig(output_path)
plt.close()

ax2.imshow(mask_image[:,:,slice2show], cmap='gray')
ax2.set_title('GT')
ax2.set_axis_off()
output_filename = 'mask_image_slice_31.png'
output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
plt.savefig(output_path)
plt.close()

dwi_image=scaler.fit_transform(dwi_image.reshape(-1, dwi_image.shape[-1])).reshape(dwi_image.shape)

X = np.zeros((72,112,112,1))
for j in range(72):
    X[j,:,:,0] = dwi_image[:,:,j]
print("X.shape: ", X.shape)

pred_wt = model.predict(X)

fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(pred_wt[31,:,:,:], cmap='gray')
output_filename = 'pred_wt_slice_31.png'
output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
plt.savefig(output_path)
plt.close()

y_pred_thresholded = pred_wt > 0.1

fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(y_pred_thresholded[31,:,:,:], cmap='gray')
output_filename = 'y_pred_thresholded_slice_31.png'
output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
plt.savefig(output_path)
plt.close()

for i in range(5,60):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(dwi_image[:,:,i], cmap='gray')
    plt.title('Input')
    plt.subplot(1, 4, 2)
    plt.imshow(mask_image[:,:,i], cmap='gray')
    plt.title('Ground Truth')
    plt.subplot(1, 4, 3)
    plt.imshow(pred_wt[i,:,:,:], cmap='gray')
    plt.title('Predicted')
    plt.subplot(1, 4, 4)
    plt.imshow(y_pred_thresholded[i,:,:,:], cmap='gray')
    plt.title('Threshold')
    dice = dice_score(mask_image[:,:,i], y_pred_thresholded[i,:,:,:])
    Iou = iou(mask_image[:,:,i], y_pred_thresholded[i,:,:,:])
    plt.suptitle(f"Sample_19_Slice_00{i}  ,Dice Score:{dice}  ,IOU:{Iou}")
    output_filename = f'Sample_19_Slice_00{i}.png'
    output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
    plt.savefig(output_path)
    plt.close()
