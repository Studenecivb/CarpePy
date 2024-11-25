import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib
from itertools import groupby

import numpy as np
from scipy.interpolate import interp1d

import math
from functools import reduce
import matplotlib.colors as mcolors

from .diem_helper_functions import *
from .mathematica2python import *


def polarise_n_join(polarisation_data, s_data):
    modified_data = []
    for i in range(len(polarisation_data)):
        polarisation_array = polarisation_data[i]
        s_column = np.array(s_data[i])[:, np.newaxis]
        new_array = np.hstack((polarisation_array, s_column))
        for row in new_array:
            if float(row[0]) ==2:
                row[-1] = StringReplace20(row[-1])
            else:
                row[-1] = row[-1]
        modified_data.append(new_array)
    return modified_data


def generate_bed_data(dimension_lengths):
    data = []
    for c in range(1, len(dimension_lengths) + 1):
        length = dimension_lengths[c - 1]
        # Create a list of tuples (compartment index, range values)
        compartment_data = [(c, value) for value in 100 * np.arange(1, length + 1)]
        data.extend(compartment_data)
    return data


# Function to split the list into chunks
def split_list(data_list, lengths):
    result = []
    start = 0
    for length in lengths:
        end = start + length
        result.append(data_list[start:end])
        start = end
    return result


def chr_mb_ticks(sgl, offset=0, delta=10**6):
    if isinstance(sgl[0], tuple):
        Mb = [x[1] for x in sgl]
    else:
        Mb = sgl
        Mb = Mb.astype(int)
    sites = offset + np.arange(1, len(sgl) + 1)
    Mbticks = np.arange(np.ceil(min(Mb) / delta), np.floor(max(Mb) / delta) + 1)
    Mb_sites_pairs = np.column_stack((Mb, sites))
    Mb_sites_pairs = Mb_sites_pairs[np.lexsort((Mb_sites_pairs[:, 1],))]
    interp_func = interp1d(Mb_sites_pairs[:, 0], Mb_sites_pairs[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    tick_positions = np.round(interp_func(Mbticks * delta)).astype(int)
    tick_values = Mbticks * delta / 10**6
    return np.column_stack((tick_values, tick_positions))

def mb_ticks(gl, delta=10**6):
    chrgl = [list(group) for _, group in pd.groupby(gl, key=lambda x: x[0])]
    lengths = [len(c) for c in chrgl]
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    ticks = [chr_mb_ticks(chrgl[i], offset=offsets[i], delta=delta) for i in range(len(chrgl))]
    return ticks


def mb1_ticks(gl):
    return mb_ticks(gl, delta=10**6)

def mb2_ticks(gl):
    return mb_ticks(gl, delta=2 * 10**6)


def append_new_column(polarisation_data, COMPdummyBEDfiles):
    modified_data = []
    for i in range(len(polarisation_data)):
        polarisation_array = polarisation_data[i]
        comp_column = np.array(COMPdummyBEDfiles[i])[:, np.newaxis]
        new_array = np.hstack((polarisation_array, comp_column))
        modified_data.append(new_array)

    return modified_data

def filter_by_threshold(polarisation_data, threshold):
    filtered_data = []
    for array in polarisation_data:
        # Convert the second column to float and filter based on the threshold
        column_to_filter = array[:, 1].astype(float)
        filtered_array = array[column_to_filter > threshold]
        filtered_data.append(filtered_array)
    return filtered_data

def transpose_data(polarisation_data):
    processed_data = []
    for array in polarisation_data:
        columns_of_interest = array[:, [-1, -2]]
        transposed_array = columns_of_interest.T
        processed_data.append(transposed_array)
    return processed_data


def BrenthisDiagnosticSiteData(BrenthisPolarisedCompartment, COMPdummyBEDfiles, BrenthisNaturalThreshold=None):
    # Add BED file information
    added_bed = append_new_column(BrenthisPolarisedCompartment, COMPdummyBEDfiles)
    # Filter by Threshold
    filtered_data = filter_by_threshold(added_bed, BrenthisNaturalThreshold)
    # Only choose columns of interest
    transposed = transpose_data(filtered_data)
    return transposed


def diem_singleton_replace(s):
    replacement_rules = {
    "__U__": "_____",
    "__0__": "_____",
    "__1__": "_____",
    "__2__": "_____",
    "UU_UU": "UUUUU",
    "UU0UU": "UUUUU",
    "UU1UU": "UUUUU",
    "UU2UU": "UUUUU",
    "00_00": "00000",
    "00U00": "00000",
    "00100": "00000",
    "00200": "00000",
    "11_11": "11111",
    "11U11": "11111",
    "11011": "11111",
    "11211": "11111",
    "22_22": "22222",
    "22U22": "22222",
    "22022": "22222",
    "22122": "22222"
}
    for old, new in replacement_rules.items():
        s = s.replace(old, new)
    return s


def transform_strings(array_list):
    transformed_arrays = []

    for array in array_list:
        string_list = array.tolist()
        trimmed_list = [s[1:] for s in string_list]
        num_chars = len(trimmed_list[0])
        columns = ['' for _ in range(num_chars)]
        for string in trimmed_list:
            for i, char in enumerate(string):
                columns[i] += char
        transformed_arrays.append(np.array(columns))

    return transformed_arrays


def apply_replacement_to_list(list_of_lists):
    result = []
    for sublist in list_of_lists:
        transformed_sublist = [diem_singleton_replace(s) for s in sublist]
        result.append(transformed_sublist)
    return result


def BrenthisDiemPlotData(BrenthisDiagnosticSiteData_res):
    site_strings = [sublist[1] for sublist in BrenthisDiagnosticSiteData_res]
    individual = [StringTranspose(sublist) for sublist in site_strings]
    without_s = [sublist[1:] for sublist in individual]
    replace_individual = apply_replacement_to_list(without_s)
    return replace_individual


def sum_corresponding_lists(state_count_lists):
    num_lists = len(state_count_lists[0])
    # Initialize cumulative sums with zeros
    cumulative_sums = [np.zeros_like(state_count_lists[0][0]) for _ in range(num_lists)]

    # Iterate through each compartment
    for comp in state_count_lists:
        for i, arr in enumerate(comp):
            cumulative_sums[i] += arr
    return cumulative_sums


def BrenthisDiemPlotDataHIs(BrenthisDiemPlotData_res):
    for_all_comps = []
    for comp in BrenthisDiemPlotData_res:
        resultie = Map(csStateCount, comp)
        for_all_comps.append(resultie)

    final_sum = sum_corresponding_lists(for_all_comps)

    his = []
    for indie in final_sum:
        his.append(pHetErrOnStateCount(indie)[0])
        # print(pHetErrOnStateCount(indie)[0])

    return his


def BrenthisDiemPlotDataHIs_order(BrenthisDiemPlotDataHIs_res, BrenthisDiemPlotData_res, dummyBrenthisNames):
    his_values = BrenthisDiemPlotDataHIs_res
    sorted_indices = sorted(range(len(his_values)), key=lambda i: his_values[i])
    BrenthisSortedNames = [dummyBrenthisNames[i] for i in sorted_indices]

    final_sorted_data = []
    for comp in BrenthisDiemPlotData_res:
        sorted_diem_data = [comp[i] for i in sorted_indices]
        final_sorted_data.append(sorted_diem_data)

    return (final_sorted_data, BrenthisSortedNames)


diemColours = [
    'white',
    mcolors.to_hex((128/255, 0, 128/255)),  # RGBColor[128/255, 0, 128/255] - Purple
    mcolors.to_hex((255/255, 229/255, 0)),  # RGBColor[255/255, 229/255, 0] - Yellow
    mcolors.to_hex((0, 128/255, 128/255))   # RGBColor[0, 128/255, 128/255] - Teal
]
char_to_index = {
    '_': 0,
    'U': 0,
    '0': 1,
    '1': 2,
    '2': 3
}


def rectangular_graph(chromosome_data, index, names_list, bed_data, path=None):
    print(f'Preparing graph data for index: {index}')
    cmap = mcolors.ListedColormap(diemColours)
    grids = []
    grid_heights = []
    for i in range(len(chromosome_data)):
        current_indie = chromosome_data[i]
        char_array = np.array([current_indie])
        index_array = [char_to_index.get(char) for char in char_array[0]]
        grid_array = np.tile(index_array, (1000, 1))
        grids.append(grid_array)
        grid_heights.append(grid_array.shape[0])
    combined_grid = np.vstack(grids)
    x_ticks = chr_mb_ticks(bed_data).astype(int)
    x_ticks_positions = x_ticks[:, 1]
    x_ticks_labels = x_ticks[:, 0]
    y_ticks_positions = np.cumsum([0] + grid_heights[:-1])
    y_tick_labels = names_list
    plt.imshow(combined_grid, cmap=cmap)
    plt.title(f"chromosome_{index}")
    plt.yticks(ticks=y_ticks_positions, labels=y_tick_labels, fontsize=6)
    plt.xticks(ticks=x_ticks_positions, labels=x_ticks_labels)
    if path:
        plt.savefig(f'{path}/chromosome_{index}.png', format='png', dpi=300)
    else:
        plt.savefig(f'chromosome_{index}.png', format='png', dpi=300)
    # plt.show()


def rectangular_graph_for_all_chromosome(chromosome_data_list, names_list, bed_data_list, path=None):
    for index, chromosome_data in enumerate(chromosome_data_list):
        index = index
        bed_data = bed_data_list[index]
        print(f'Preparing graph data for index: {index}')
        cmap = mcolors.ListedColormap(diemColours)
        grids = []
        grid_heights = []
        for i in range(len(chromosome_data)):
            current_indie = chromosome_data[i]
            char_array = np.array([current_indie])
            index_array = [char_to_index.get(char) for char in char_array[0]]
            grid_array = np.tile(index_array, (1000, 1))
            grids.append(grid_array)
            grid_heights.append(grid_array.shape[0])
        combined_grid = np.vstack(grids)
        x_ticks = chr_mb_ticks(bed_data).astype(int)
        x_ticks_positions = x_ticks[:, 1]
        x_ticks_labels = x_ticks[:, 0]
        y_ticks_positions = np.cumsum([0] + grid_heights[:-1])
        y_tick_labels = names_list
        plt.imshow(combined_grid, cmap=cmap)
        plt.title(f"chromosome_{index}")
        plt.yticks(ticks=y_ticks_positions, labels=y_tick_labels, fontsize=6)
        plt.xticks(ticks=x_ticks_positions, labels=x_ticks_labels)
        if path:
            plt.savefig(f'{path}/chromosome_{index}.png', format='png', dpi=300)
        else:
            plt.savefig(f'chromosome_{index}.png', format='png', dpi=300)
        # plt.show()

#
# def circular_graph(chromosome_data_list, names_list, bed_data_list, path=None):
#     index = 0
#     bed_data = bed_data_list[index]
#     print(f'Preparing graph data for index: {index}')
#
#     # Set up color map
#     cmap = mcolors.ListedColormap(diemColours)
#
#     # Prepare grids and their heights
#     grids = []
#     grid_heights = []
#     for i in range(len(chromosome_data_list[0])):
#         current_indie = chromosome_data_list[0][i]
#         char_array = np.array([current_indie])
#         index_array = [char_to_index.get(char) for char in char_array[0]]
#         grid_array = np.tile(index_array, (500, 1)).astype(np.float16)  # Use float32 to reduce memory
#         grids.append(grid_array)
#         grid_heights.append(grid_array.shape[0])
#
#     # Combine grids and ensure data type is float32 or smaller
#     combined_grid = np.vstack(grids).astype(np.float16)  # or use np.float16 for further reduction
#
#     # Prepare polar coordinates for the circular graph
#     num_rows, num_cols = combined_grid.shape
#     theta = np.linspace(0, 2 * np.pi, num_cols).astype(np.float16)
#     r = np.linspace(0, 1, num_rows).astype(np.float16)
#
#     # Create meshgrid for polar coordinates
#     Theta, R = np.meshgrid(theta, r)
#
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#     ax.pcolormesh(Theta, R, combined_grid, cmap=cmap, shading='auto')
#
#     ax.set_title(f"chromosome_{index}")
#
#     plt.show()
#
#     # Optionally save the figure
#     if path:
#         plt.savefig(f'{path}/chromosome_{index}_circular.jpg', format='jpg', dpi=50)
#     else:
#         plt.savefig(f'chromosome_{index}_circular.png', format='jpg', dpi=50)


# Run Lenght Compression bit:

def TsRishRLEjoin(a, b):
    # Should be good
    if a[-1][0] == b[0][0]:
        return a[:-1] + [[a[-1][0], a[-1][1] + b[0][1], a[-1][2], b[0][3]]] + b[1:]
    else:
        return a + b


def Last(lst):
    return lst[-1]


def sRichRLEmerger(ls):
    endofprevious = Drop((Join([0], Accumulate(Map(Last, Map(Last, ls))))), -1)
    Tls = [
        list(zip(
            ls[i][0],  # First column
            ls[i][1],  # Second column
            [x + endofprevious[i] for x in ls[i][2]],  # Third column adjusted by endofprevious
            [x + endofprevious[i] for x in ls[i][3]]  # Fourth column adjusted by endofprevious
        ))
        for i in range(len(ls))]

    merged = Tls[0]
    for i in range(1, len(Tls)):
        merged = TsRishRLEjoin(merged, Tls[i])
    merged_transposed = list(zip(*merged))
    return merged_transposed


def rle_brenthis_plot_data(diem_plot_data):
    rle_step = []
    for comp in diem_plot_data:
        comp_rle = Map(RichRLE, comp)
        rle_step.append(comp_rle)
        # [states, lengths, starts, ends]
    transposed_all = list(map(list, zip(*rle_step)))
    merged_all = Map(sRichRLEmerger, transposed_all)
    final_result = Map(BackgroundedRLE, merged_all)
    return final_result


def flatten1(lst):
    return [item for sublist in lst for item in sublist]


def rle_brenthis_diem_plot_bed(BrenthisDiemPlotBED):
    flattened_bed = flatten1(BrenthisDiemPlotBED)
    first_elements = [x for x in flattened_bed]
    rle_result = RLE(first_elements)
    rle_transposed = np.array(rle_result).T
    lengths_list = [len(arr) for arr in BrenthisDiemPlotBED]
    split_indices = np.cumsum(lengths_list[:-1])
    split_arrays = np.split(rle_transposed, split_indices)
    return split_arrays


def rle_brenthis_sorted(BrenthisDiemPlotDataHIs_res, rleBrenthisPlotData_res):
    his_values = BrenthisDiemPlotDataHIs_res
    sorted_indices = sorted(range(len(his_values)), key=lambda i: his_values[i])
    final_sorted_data = [rleBrenthisPlotData_res[i] for i in sorted_indices]

    return final_sorted_data


from matplotlib.patches import Wedge
import matplotlib.colors as mcolors


class WheelDiagram:
    def __init__(self, subplot, center, radius, number_of_rings, cutout_angle=13):
        self.subplot = subplot
        self.center = center
        self.radius = radius
        self.center_radius = radius / 2
        self.number_of_rings = number_of_rings
        self.cutout_angle = cutout_angle
        self.rings_added = 0

    def add_wedge(self, radius, from_angle, to_angle, color):
        self.subplot.add_artist(
            Wedge(self.center, radius, from_angle, to_angle, color=color)
        )

    def add_ring(self, list_of_thingies):
        available_angle = 360 - self.cutout_angle
        angle_scale = available_angle / list_of_thingies[-1][-1]
        color_map = {
            "_": "white",
            "U": "white",
            "0": mcolors.to_hex((128 / 255, 0, 128 / 255)),
            "1": mcolors.to_hex((255 / 255, 229 / 255, 0)),
            "2": mcolors.to_hex((0, 128 / 255, 128 / 255))
        }
        ring_radius = self.radius - self.rings_added * (self.radius - self.center_radius) / self.number_of_rings

        start_angle_offset = 90
        for index, thing in enumerate(list_of_thingies):
            from_angle = start_angle_offset + 360 - (angle_scale * (thing[1] - 1))
            to_angle = start_angle_offset + 360 - (angle_scale * thing[2])
            self.add_wedge(ring_radius, to_angle,from_angle, color_map[thing[0]])

        self.rings_added += 1

    def clear_center(self):
        self.add_wedge(self.center_radius, 0, 360, "white")


def ring_rle(input_data, path, names=None, pdf=None, png=None, bed_info=None, length_of_chromosomes=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_aspect('equal')
    center = (.5, .5)
    radius = 0.5
    number_of_rings = len(input_data)
    cutout_angle = 14

    wd = WheelDiagram(ax, center, radius, number_of_rings, cutout_angle=cutout_angle)
    for ring in input_data:
        wd.add_ring(ring)
    wd.clear_center()

    outer_radius = radius + 0.01  # Adjust this for better label distance
    available_angle = 360 - cutout_angle
    start_angle_offset = 90

    if length_of_chromosomes:
        max_position = input_data[0][-1][2]
        inner_radius = outer_radius - 0.05

        # The inner labels
        for idx, (label, (start, end, length)) in enumerate(length_of_chromosomes.items()):
            # Calculate the angles for the start and end of each chromosome
            start_angle = start_angle_offset + 360 - (available_angle * start / max_position)
            end_angle = start_angle_offset + 360 - (available_angle * end / max_position)

            # Add alternating shading
            color = 'lightgrey' if idx % 2 == 0 else 'none'  # Alternate color
            ax.add_artist(Wedge(center, radius/2, end_angle, start_angle, color=color, alpha=0.3))
            ax.add_artist(Wedge(center, 0.18, 0, 360, color='white'))

            # Draw lines for start and end positions
            for angle in [start_angle, end_angle]:
                angle_rad = np.radians(angle)

                # Calculate coordinates for the lines, move inward
                line_start_x = center[0] + (outer_radius - 0.28) * np.cos(angle_rad)
                line_start_y = center[1] + (outer_radius - 0.28) * np.sin(angle_rad)

                # Draw the line
                ax.plot([line_start_x, line_start_x + 0.02 * np.cos(angle_rad)],
                        [line_start_y, line_start_y + 0.02 * np.sin(angle_rad)],
                        color='black', linewidth=0.5)

            # Calculate the midpoint for placing the label
            midpoint = (start + end) / 2
            midpoint_angle = start_angle_offset + 360 - (available_angle * midpoint / max_position)
            midpoint_angle_rad = np.radians(midpoint_angle)

            # Calculate position for the label, move inward
            label_x = center[0] + (outer_radius - 0.3) * np.cos(midpoint_angle_rad)
            label_y = center[1] + (outer_radius - 0.3) * np.sin(midpoint_angle_rad)

            # Place the label
            ax.text(label_x, label_y, str(label), ha='center', va='center', fontsize=6)

    # The outer ticks
    if bed_info is not None:
        # Maximum position for scaling
        max_position = input_data[0][-1][2]
        for chrom, positions in bed_info.items():
            label_counter = 1  # Initialize a counter
            for label, position in positions:
                angle = start_angle_offset + 360 - (available_angle * position / max_position)
                angle_rad = np.radians(angle)

                x = center[0] + outer_radius * np.cos(angle_rad)
                y = center[1] + outer_radius * np.sin(angle_rad)

                if label_counter % 2 == 0:
                    label_distance = 0.005
                    label_x = x + label_distance * np.cos(angle_rad)
                    label_y = y + label_distance * np.sin(angle_rad)

                    ax.text(label_x, label_y, str(label), ha='center', va='bottom', fontsize=6, rotation=angle - 90,
                            rotation_mode='anchor')

                    inward_offset = 0.01
                    line_start_x = x - inward_offset * np.cos(angle_rad)
                    line_start_y = y - inward_offset * np.sin(angle_rad)

                    line_length = 0.01
                    ax.plot([line_start_x, line_start_x + line_length * np.cos(angle_rad)],
                            [line_start_y, line_start_y + line_length * np.sin(angle_rad)],
                            color='black', linewidth=0.5)  # Draw the line

                label_counter += 1

    # Chromosome names
    if names and len(names) == number_of_rings:
        for i, name in enumerate(names):
            ring_radius = radius - (i + 0.5) * (radius - wd.center_radius) / number_of_rings

            label_angle = 90
            angle_rad = np.radians(label_angle)
            label_x = center[0] + ring_radius * np.cos(angle_rad)
            label_y = center[1] + ring_radius * np.sin(angle_rad)
            ax.text(label_x, label_y, name, ha='right', va='center', fontsize=10, rotation=label_angle - 90,
                    rotation_mode='anchor')

    # Save the output files
    if png:
        plt.savefig(f'{path}/{png}.png', format="png")
    if pdf:
        plt.savefig(f'{path}/{pdf}.pdf', format="pdf")

    if not pdf and not png:
        plt.show()  # Display the plot if desired















