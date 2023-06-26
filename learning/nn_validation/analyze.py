import numpy as np
import matplotlib.pyplot as plt

# load data.csv
data = np.loadtxt('data.csv', delimiter=',')

#N = 10000
N = data.shape[0]
input_all = data[:N, :10]
nn_dist_all = data[:N, 10:19]
mesh_dist_all = data[:N, 19:]

print("%d samples loaded" % input_all.shape[0])
# remove negative distances
thr = 3
mindist_nn = nn_dist_all.min(axis=1)
idx_mask = mindist_nn > thr
input = input_all[idx_mask, :]
nn_dist = nn_dist_all[idx_mask, :]
mesh_dist = mesh_dist_all[idx_mask, :]
print("%d samples remain after filtering" % input.shape[0])

n_links = 9

# #plot distributions of distances
# fig = plt.figure()
# for i in range(n_links):
#     ax = fig.add_subplot(3, 3, i+1)
#     #ax.hist(nn_dist[:, i], bins=100, label='NN')
#     #ax.hist(mesh_dist[:, i], bins=100, label='Mesh')
#     L1 = np.abs(nn_dist[:, i] - mesh_dist[:, i])
#     ax.hist(mesh_dist[:, i], bins=100, label='Mean L1 error: %4.2f\nMax L1 error: %4.2f'%(L1.mean(), L1.max()))
#     ax.legend(loc='upper right')
#     ax.title.set_text('Link %d' % (i+1))
#     print('Link %d: L1 error: %4.2f, Max err: %4.2f' % (i, L1.mean(), L1.max()))
# fig.set_size_inches(18.5, 10.5)
# plt.savefig('dist_distrib.png', dpi=600, bbox_inches='tight')
# plt.show(block=False)
plt.rcParams.update({'font.size': 12})
fig = plt.figure()
# Define error bins
bin_w = 20
bins = [i for i in range(thr, 150, bin_w)] # 3cm bins
all_outliers = []
for i in range(n_links):
    ax = fig.add_subplot(3, 3, i+1)
    L1 = np.abs(nn_dist[:, i] - mesh_dist[:, i])


    # Compute mean and max L1 errors for each bin
    bin_errors = []
    mean_errors = []
    max_errors = []
    bin_fliers = []
    bin_all_outliers = []
    for j in range(len(bins)-1):
        # Select errors in current bin
        idx_bin = (mesh_dist[:, i] >= bins[j]) & (mesh_dist[:, i] < bins[j+1])
        errors = L1[idx_bin]
        inputs_lcl = input[idx_bin, :]
        mesh_dist_lcl = mesh_dist[idx_bin, :]
        nn_dist_lcl = nn_dist[idx_bin, :]
        # Compute and store mean and max error
        mean_errors.append(errors.mean())
        max_errors.append(errors.max())


        # Compute quartiles and interquartile range
        q1, q3 = np.percentile(errors, [25, 75])
        iqr = q3 - q1
        # Identify outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (errors < lower_bound) | (errors > upper_bound)
        outlier_indices = np.where(outlier_mask)[0]
        bin_outliers_raw = np.concatenate([inputs_lcl[outlier_indices, :], nn_dist_lcl[outlier_indices, :], mesh_dist_lcl[outlier_indices, :]], axis=1)
        bin_all_outliers.append(bin_outliers_raw)
        n_outliers = 20
        outlier_indices_ord = np.argsort(errors[outlier_indices])
        outlier_indices_sorted = outlier_indices[outlier_indices_ord]
        # Subsample outliers if there are more than 10
        if len(outlier_indices) > n_outliers:
            step = len(outlier_indices_sorted) // (n_outliers)
            subsample_indices = outlier_indices_sorted[::-1][::step][:(n_outliers // 2)]
            subsample_indices = np.append(subsample_indices, outlier_indices_sorted[::-1][:100:10])
            subsample_indices = np.unique(subsample_indices)
            # Create a new mask that only includes the subsampled outliers
            new_mask = np.full(errors.shape, False)
            new_mask[subsample_indices] = True

            # Use this new mask to keep only the subsampled outliers and remove the others
            #errors = errors[~outlier_mask | new_mask]
            fliers = errors[new_mask]
        bin_errors.append(errors)
        bin_fliers.append(fliers)
    all_outliers.append(np.concatenate(bin_all_outliers, axis=0))
    # # Plot mean and max errors
    # ax.bar(bins[:-1], max_errors, width=bin_w, color=[0.8500, 0.3250, 0.0980], alpha=0.9, label='Max L1 error (avg: %4.2f cm)'%L1.max())
    # ax.bar(bins[:-1], mean_errors, width=bin_w, color=[0, 0.4470, 0.7410], alpha=0.9, label='Mean L1 error (avg: %4.2f cm)'%L1.mean())
    ax.boxplot(bin_errors, positions=np.array(bins[:-1]), widths=bin_w-5, showmeans=False,
               flierprops={'marker': '+', 'color': 'gray', 'alpha': 0.5}, showfliers=False)
    # Overlay outliers
    for j, outliers in enumerate(bin_fliers):
        ax.plot([bins[j]] * len(outliers), outliers, '+', color='gray', alpha=0.5)

    ax.set_title(f'Link {i + 1}')
    bin_labels = [f'{int(bins[j])}-{int(bins[j + 1])}' for j in range(len(bins) - 1)]
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels(bin_labels, rotation=45)

    if i in [0, 3, 6]:
        ax.set_ylabel('L1 Error, cm', fontsize=14)
    if i >=6:
        ax.set_xlabel('Distance, cm. Bin width = 20 cm.', fontsize=14)
    if i in [0, 1, 2]:
        ax.set_ylim([0, 4])  # Set y-axis range to 0-4cm
    elif i in [6, 7, 8]:
        ax.set_ylim([0, 16])  # Set y-axis range to 0-16cm

    #ax.legend(loc='upper right')
    print('Link %d: L1 error: %4.2f, Max err: %4.2f' % (i, L1.mean(), L1.max()))
plt.subplots_adjust(hspace=0.4)
fig.set_size_inches(15, 15)

plt.savefig('dist_barplot.png', dpi=300)#, bbox_inches='tight')
plt.show(block=True)


# # save data with errors above threshold
# all_err = abs(nn_dist_all - mesh_dist_all)
# idx_large_err = np.any(all_err > 5, 1)
# print(data[idx_large_err * idx_mask, :].shape)
# print('Large errors: %d' % idx_large_err.sum())
# np.savetxt('data_large_err.csv', data[idx_large_err * idx_mask, :], delimiter=',', fmt='%4.3f')

# # save outliers
for i, data in enumerate(all_outliers):
    np.savetxt('outliers_link/outliers_link%d.csv' % (i+1), data, delimiter=',', fmt='%4.3f')