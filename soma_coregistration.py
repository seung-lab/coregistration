from cloudvolume import CloudVolume
from cloudvolume.lib import min2, Vec, Bbox, mkdir
import numpy as np

image_resolution = Vec(4, 4, 40)
offset = Vec(-3072, -2560, 7900)

# 8x8x40 mip
def get_cloudvolume_coordinates_vector(nano_vec):
    mip4_4_40_vec = nano_vec // image_resolution + offset
    return Vec(mip4_4_40_vec[0] // 2, mip4_4_40_vec[1] // 2, mip4_4_40_vec[2])


# 4x4x40 mip
def get_neuroglancer_coordinates_vector(nano_vec):
    return nano_vec // image_resolution + offset


cv = CloudVolume(
    "gs://microns-seunglab/minnie65/seg_minnie65_0", bounded=False, autocrop=True
)


def output_line_with_segment_id(original_line, seg_id, neuroglancer_vec):
    return (
        original_line.strip()
        + ",,"
        + str(seg_id)
        + ","
        + str(neuroglancer_vec[0])
        + ","
        + str(neuroglancer_vec[1])
        + ","
        + str(neuroglancer_vec[2])
        + "\n"
    )


def output_line_without_segment_id(original_line, neuroglancer_vec):
    return (
        original_line.strip()
        + ",,"
        + str(neuroglancer_vec[0])
        + ","
        + str(neuroglancer_vec[1])
        + ","
        + str(neuroglancer_vec[2])
        + "\n"
    )


def simple_categorization(
    input_file,
    categorized_output_file,
    uncategorized_output_file,
    threshold=0.5,
    mip=5,
    box_sample=(8, 8, 32),
):
    """
    For each coordinate in the input_file, find the most common occurring segment in the box_sample 
    bounding box with the coordinate at its center at the mip specified. If this segment appears 
    in a proportion equal to or greater than threshold (threshold should bebetween 0 and 1), the 
    coordinate and segment get written to the categorized_output_file. Otherwise, the coordinate gets 
    written to the uncategorized_output_file. The default parameters are the ones I used in the initial run.
    """
    i = 1
    with open(input_file, "r") as input_f:
        with open(categorized_output_file, "w") as output_cat, open(
            uncategorized_output_file, "w"
        ) as output_uncat:
            line = input_f.readline()
            while line:
                if i % 100 == 0:
                    print("i = ", i)
                elements = line.split(",")
                nano_vec = Vec(
                    int(float(elements[0].strip())),
                    int(float(elements[1].strip())),
                    int(float(elements[2].strip())),
                )
                cloudvolume_vec = get_cloudvolume_coordinates_vector(nano_vec)
                # Include neuroglancer coordinates in the output for easy checking and debugging
                neuroglancer_vec = get_neuroglancer_coordinates_vector(nano_vec)
                cutout = cv.download_point(cloudvolume_vec, size=box_sample, mip=mip)
                unique_count = np.unique(cutout, return_counts=True)
                max_index = np.argmax(unique_count[1])
                seg_id = unique_count[0][max_index]
                if seg_id == 0:
                    # Darkness surrounds us...
                    output_uncat.write(
                        output_line_without_segment_id(line, neuroglancer_vec)
                    )
                elif unique_count[1][max_index] >= np.prod(box_sample) * threshold:
                    output_cat.write(
                        output_line_with_segment_id(line, seg_id, neuroglancer_vec)
                    )
                else:
                    output_uncat.write(
                        output_line_without_segment_id(line, neuroglancer_vec)
                    )
                line = input_f.readline()
                i = i + 1


def second_filter_categorization(
    input_file,
    categorized_output_file,
    uncategorized_output_file,
    minimum_voxel_count=15000,
    top_segment_minimum_multiplier=1.5,
    mip=5,
    box_sample=(32, 32, 128),
):
    """
    For each coordinate in the input_file, find the most common occurring segment in the box_sample 
    bounding box with the coordinate at its center at the mip specified. If this segment appears at 
    least minimum_voxel_count times, and at least top_segment_minimum_multiplier times more than the
    second most occurring segment (ignoring segment 0), then the coordinate and segment get written 
    to the categorized_output_file. Otherwise, the coordinate gets written to the 
    uncategorized_output_file. The default parameters are the ones I used in the initial run.
    """
    i = 0
    with open(input_file, "r") as input_f:
        with open(categorized_output_file, "w") as output_cat, open(
            uncategorized_output_file, "w"
        ) as output_uncat:
            line = input_f.readline()
            while line:
                i = i + 1
                if i % 100 == 0:
                    # Takes a while to run (~100coordinates/min) so this is useful
                    print("i = ", i)
                elements = line.split(",")
                nano_vec = Vec(
                    int(float(elements[0].strip())),
                    int(float(elements[1].strip())),
                    int(float(elements[2].strip())),
                )
                cloudvolume_vec = get_cloudvolume_coordinates_vector(nano_vec)
                neuroglancer_vec = get_neuroglancer_coordinates_vector(nano_vec)
                cutout = cv.download_point(cloudvolume_vec, size=box_sample, mip=mip)
                unique_count = np.unique(cutout, return_counts=True)
                if len(unique_count[0]) == 0:
                    output_uncat.write(
                        output_line_without_segment_id(line, neuroglancer_vec)
                    )
                else:
                    zero_ind = np.where(unique_count[0] == 0)
                    # Delete 0 to ignore it
                    if len(zero_ind[0]) > 0:
                        unique_count = (
                            np.delete(unique_count[0], zero_ind[0][0]),
                            np.delete(unique_count[1], zero_ind[0][0]),
                        )
                    if len(unique_count[0]) == 0:
                        output_uncat.write(
                            output_line_without_segment_id(line, neuroglancer_vec)
                        )
                    else:
                        max_index = np.argmax(unique_count[1])
                        seg_id = unique_count[0][max_index]
                        if unique_count[1][max_index] < minimum_voxel_count:
                            output_uncat.write(
                                output_line_without_segment_id(line, neuroglancer_vec)
                            )
                        else:
                            second_seg_count = 0
                            if len(unique_count[0]) > 1:
                                # Find 2nd most occurring segment
                                partition = np.argpartition(unique_count[1], -2)[-2:]
                                for x in range(2):
                                    if partition[x] != max_index:
                                        second_most_ind = partition[x]
                                second_seg_count = unique_count[1][second_most_ind]
                            seg_count = unique_count[1][max_index]
                            if (
                                seg_count
                                >= top_segment_minimum_multiplier * second_seg_count
                            ):
                                seg_id = unique_count[0][max_index]
                                output_cat.write(
                                    output_line_with_segment_id(
                                        line, seg_id, neuroglancer_vec
                                    )
                                )
                            else:
                                output_uncat.write(
                                    output_line_without_segment_id(
                                        line, neuroglancer_vec
                                    )
                                )
                line = input_f.readline()


# Examples
# simple_categorization('Allen_proofread_mm_with_em.csv', 'Allen_test_cat.csv', 'Allen_test_uncat.csv')
second_filter_categorization(
    "baylor_mm_with_em.csv", "baylor_test_cat.csv", "baylor_test_uncat.csv"
)
