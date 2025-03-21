from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from fractions import Fraction
import numpy as np

from matplotlib.patches import Wedge
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
import seaborn as sns

from .mathematica2python import StringTranspose, RichRLE, Flatten, StringTakeList, Map
from .diem_helper_functions import StringReplace20, pHetErrOnString, sStateCount
from .kernel_smoothing import n_laplace_smooth_one_haplotype

import matplotlib
matplotlib.use('Agg')

# Polarise and Join
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


# AnnotatedHITally
def AnnotatedHITally(markers):
    string_counts = Counter(markers)
    sorted_counts = sorted(string_counts.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sorted_counts, columns=["Type", "N"])
    total_strings = len(markers)
    df["p"] = df["N"] / total_strings
    df["cum(p)"] = df["p"].cumsum()
    return df


# DiemPlotPrep Class
class DiemPlotPrep:
    def __init__(self, plot_theme, ind_ids, polarised_data, di_threshold, di_column, phys_res, ticks=None, smooth=None):
        self.polarised_data = polarised_data
        self.di_threshold = di_threshold
        self.di_column = di_column
        self.phys_res = phys_res
        self.plot_theme = plot_theme
        self.ind_ids = ind_ids
        self.ticks = ticks
        self.smooth = smooth

        self.diemPlotLabel = None
        self.DIfilteredDATA = None
        self.DIfilteredGenomes = None
        self.DIfilteredHIs = None
        self.DIfilteredBED = None
        self.DIpercent = None
        self.DIfilteredScafRLEs = None
        self.diemDITgenomes = None
        self.DIfilteredGenomes_unsmoothed = None
        self.DIfilteredBED_formatted = None
        self.IndIDs_ordered = None
        self.unit_plot_prep = []
        self.plot_ordered = None
        self.length_of_chromosomes = {}
        self.iris_plot_prep = {}
        self.diemDITgenomes_ordered = None

        self.diem_plot_prep()

    def diem_plot_prep(self):
        """ Perform DI filtering, dithering, and label generation """
        self.filter_data()
        if self.smooth:
            self.kernel_smooth(self.smooth)
        self.diem_dithering()

        self.generate_plot_label(self.plot_theme)

    def format_bed_data(self):
        grouped = {}
        for item in self.DIfilteredBED:
            key, value = item
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(value)

        self.DIfilteredBED_formatted = [np.array(values) for values in grouped.values()]

        self.plot_ordered = self.DIfilteredHIs
        for a, b in enumerate(self.plot_ordered):
            try:
                self.plot_ordered[a] = (float(b[0]), a + 1)
            except ValueError:
                self.plot_ordered[a] = (np.nan, a + 1)
        self.plot_ordered = sorted(self.plot_ordered, key=lambda x: (np.isnan(x[0]), x[0]))
        # sort the names according to the HIs
        sorted_indices = [index - 1 for _, index in self.plot_ordered]
        # Reorder the names using the sorted indices
        self.IndIDs_ordered = [self.ind_ids[i] for i in sorted_indices]

        start_position = 0
        for bed_data in self.DIfilteredBED_formatted:
            sublist = []
            end_position = start_position + len(bed_data)
            for genome in self.DIfilteredGenomes:
                sublist.append(genome[start_position:end_position])
            sorted_sublist = [sublist[idx] for idx in sorted_indices]
            self.unit_plot_prep.append(sorted_sublist)
            start_position = end_position

        start = 0

        for index, value in enumerate(self.DIfilteredBED_formatted):
            end = start + len(value)  # Current value is the ending point
            length = len(value)  # Calculate the length
            self.length_of_chromosomes[list(grouped.keys())[index]] = [start, end, length]  # Create the dictionary entry
            start = end  # Update the starting point for the next iteration

        for index, bed in enumerate(self.DIfilteredBED_formatted):
            if self.ticks in ['kb', 'KB', 'Kb']:
                x_ticks = chr_kb_ticks(bed).astype(int)
            else:
                x_ticks = chr_mb_ticks(bed).astype(int)
            new_ticks = np.zeros_like(x_ticks)
            starting_point = self.length_of_chromosomes[list(grouped.keys())[index]][0]
            for i, item in enumerate(x_ticks):
                new_value = item[1] + starting_point
                new_ticks[i] = [item[0], new_value]
            self.iris_plot_prep[index + 1] = new_ticks
        self.diemDITgenomes_ordered = [self.diemDITgenomes[i] for i in sorted_indices]

    def filter_data(self):
        """ Apply DI threshold filtering on the data """
        if isinstance(self.di_threshold, str):  # No filtering if threshold is a string
            self.DIfilteredDATA = self.polarised_data
        elif len(self.di_threshold) == 0:  # Filter above if threshold is just one number
            self.DIfilteredDATA = [row for row in self.polarised_data if row[self.di_column] >= self.di_threshold]
        else:  # Filter within an interval
            self.DIfilteredDATA = [row for row in self.polarised_data if self.di_threshold[0] <= row[self.di_column] <= self.di_threshold[1]]

        # Extract relevant data after filtering
        self.DIfilteredGenomes = StringTranspose([row[2] for row in self.DIfilteredDATA])
        self.DIfilteredHIs = [pHetErrOnString(genome) for genome in self.DIfilteredGenomes]
        self.DIfilteredBED = [row[:2] for row in self.DIfilteredDATA]
        self.DIpercent = round(100 * len(self.DIfilteredDATA) / len(self.polarised_data))
        self.DIfilteredScafRLEs = RichRLE([row[0] for row in self.DIfilteredBED])

    def kernel_smooth(self, scale):
        from collections import defaultdict
        scaffold_indices = defaultdict(list)
        for idx, entry in enumerate(self.DIfilteredBED):
            scaffold = entry[0]
            scaffold_indices[scaffold].append(idx)

        split_genomes = []
        for genome in self.DIfilteredGenomes:
            scaffold_dict = {}
            for scaffold, indices in scaffold_indices.items():
                scaffold_str = ''.join([genome[i] for i in indices])
                scaffold_dict[scaffold] = scaffold_str
            split_genomes.append(scaffold_dict)

        scaffold_positions = defaultdict(list)
        for entry in self.DIfilteredBED:
            scaffold = entry[0]  # Scaffold name
            position = entry[1]  # Position
            scaffold_positions[scaffold].append(position)
        scaffold_arrays = {scaffold: np.array(positions) for scaffold, positions in scaffold_positions.items()}

        smoothed_split_genomes = []
        # going through individuals
        for idx, genome in enumerate(split_genomes):
            smoothed_individual_genome = {}
            for key, value in genome.items():
                cleaned_string = value.replace("_", "3")
                integer_list = [int(char) for char in cleaned_string]
                numpy_array_haplo = np.array(integer_list)
                smooth_output = n_laplace_smooth_one_haplotype(scaffold_arrays[key], numpy_array_haplo, scale)
                string_list = [str(x) for x in smooth_output]
                string_list = ['_' if x == '3' else x for x in string_list]
                result_string = ''.join(string_list)
                smoothed_individual_genome[key] = result_string
            smoothed_split_genomes.append(smoothed_individual_genome)
        self.DIfilteredGenomes_unsmoothed = self.DIfilteredGenomes
        self.DIfilteredGenomes = self._reconstruct_genomes(smoothed_split_genomes, scaffold_indices)

    def _reconstruct_genomes(self, smoothed_split_genomes, scaffold_indices):
        reconstructed_genomes = []

        for individual in smoothed_split_genomes:
            full_genome = ['0'] * len(self.DIfilteredBED)

            for scaffold, indices in scaffold_indices.items():
                scaffold_str = individual[scaffold]
                for i, idx in enumerate(indices):
                    full_genome[idx] = scaffold_str[i]

            reconstructed_genome = ''.join(full_genome)
            reconstructed_genomes.append(reconstructed_genome)

        return reconstructed_genomes


    def diem_dithering(self):
        """ Perform dithering on the filtered data """
        diem_dit_genomes_bed = [list(group) for _, group in groupby(self.DIfilteredBED, key=lambda x: x[0])]
        processed_diemDITgenomes = []
        for chr in diem_dit_genomes_bed:
            length_data = [row[1] for row in chr]
            split_lengths = self.GappedQuotientSplitLengths(length_data, self.phys_res)
            processed_diemDITgenomes.append(split_lengths)
        processed_diemDITgenomes = Flatten(processed_diemDITgenomes)
        diemDITgenomes = []
        for genome in self.DIfilteredGenomes:
            string_take_result = StringTakeList(genome, processed_diemDITgenomes)
            state_count = Map(sStateCount, string_take_result)
            combined = list(zip(state_count, processed_diemDITgenomes))
            # transposed = Transpose([state_count, processed_diemDITgenomes])
            compressed = self.DITcompress(combined)
            lengths = self.Lengths2StartEnds(compressed)
            diemDITgenomes.append(lengths)

        self.diemDITgenomes = diemDITgenomes


    def generate_plot_label(self, plot_theme):
        """ Generate the label for the plot """
        self.diemPlotLabel = f"{plot_theme} @ DI = {self.di_threshold}: {len(self.DIfilteredGenomes)} sites ({self.DIpercent}%)."

    @staticmethod
    def GappedQuotientSplit(lst, Q):
        """
        Splits the list `lst` into sublists where consecutive elements share the same quotient when divided by `Q`.
        """
        quotients = [x // Q for x in lst]

        groups = []
        current_group = [lst[0]]

        for i in range(1, len(lst)):
            if quotients[i] == quotients[i - 1]:
                current_group.append(lst[i])
            else:
                groups.append(current_group)
                current_group = [lst[i]]

        groups.append(current_group)
        return groups

    def GappedQuotientSplitLengths(self, lst, Q):
        """
        Returns the lengths of the sublists produced by `gapped_quotient_split`.
        """
        return Map(len, self.GappedQuotientSplit(lst, Q))

    @staticmethod
    def normalize_4list(lst):
        """
        Normalizes a 4list by converting each element to its ratio of the total sum.
        Uses Fraction for precise comparison without floating-point errors.
        """
        total = sum(lst)
        if total == 0:
            return tuple(0 for _ in lst)  # Handle case where total is 0
        return tuple(Fraction(x, total) for x in lst)

    def DITcompress(self, DITl):
        """
        Compresses the list of {4list, length} tuples.
        """
        grouped_data = [list(group) for _, group in groupby(DITl, key=lambda x: self.normalize_4list(x[0]))]
        final_data = []
        for group in grouped_data:
            summed_states = [sum(x) for x in zip(*(item[0] for item in group))]
            summed_value = sum(item[1] for item in group)
            result = (summed_states, summed_value)
            final_data.append(result)
        return final_data

    @staticmethod
    def Lengths2StartEnds(stateNlen):
        lengths = [x[1] for x in stateNlen]
        ends = np.cumsum(lengths)

        # Calculate the start positions (end positions minus length plus 1)
        starts = ends - np.array(lengths) + 1

        # Combine states, starts, and ends into a list of triplets
        result = [(state, int(start), int(end)) for (state, start, end) in zip([x[0] for x in stateNlen], starts, ends)]

        return result


# MB to ticks:
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


def chr_kb_ticks(sgl, offset=0, delta=10**5):
    if isinstance(sgl[0], tuple):
        Kb = [x[1] for x in sgl]
        Kb = np.array(Kb).astype(float).astype(int)
    else:
        Kb = sgl
        Kb = np.array(Kb).astype(float).astype(int)
    sites = offset + np.arange(1, len(sgl) + 1)
    Kbticks = np.arange(np.ceil(min(Kb) / delta), np.floor(max(Kb) / delta) + 1)
    Kb_sites_pairs = np.column_stack((Kb, sites))
    Kb_sites_pairs = Kb_sites_pairs[np.lexsort((Kb_sites_pairs[:, 1],))]
    interp_func = interp1d(
        Kb_sites_pairs[:, 0], Kb_sites_pairs[:, 1],
        kind='linear', bounds_error=False, fill_value="extrapolate"
    )
    tick_positions = np.round(interp_func(Kbticks * delta)).astype(int)
    tick_values = Kbticks * delta / 10 ** 3  # Convert to kilobases (kb)

    return np.column_stack((tick_values, tick_positions))


def kb_ticks(gl, delta=10 ** 3):
    chrgl = [list(group) for _, group in pd.groupby(gl, key=lambda x: x[0])]
    lengths = [len(c) for c in chrgl]
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    ticks = [chr_kb_ticks(chrgl[i], offset=offsets[i], delta=delta) for i in range(len(chrgl))]
    return ticks


def kb1_ticks(gl):
    return kb_ticks(gl, delta=10 ** 3)


def kb2_ticks(gl):
    return kb_ticks(gl, delta=2 * 10 ** 3)


# DiemRectangleDiagram
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


def diemUnitPlot(chromosome_data, index, names_list, bed_data, path=None, ticks=None):
    print(f'Preparing graph data for index: {index}')
    cmap = mcolors.ListedColormap(diemColours)
    grids = []
    grid_heights = []
    for i in range(len(chromosome_data)):
        current_indie = chromosome_data[i]
        char_array = np.array([current_indie])
        index_array = [char_to_index.get(char) for char in char_array[0]]
        grid_array = np.tile(index_array, (10, 1))
        grids.append(grid_array)
        grid_heights.append(grid_array.shape[0])
    combined_grid = np.vstack(grids)
    bed_data = bed_data.astype(str)
    if ticks == 'kb':
        x_ticks = chr_kb_ticks(bed_data).astype(int)
    else:
        x_ticks = chr_mb_ticks(bed_data).astype(int)
    x_ticks_positions = x_ticks[:, 1]
    x_ticks_labels = x_ticks[:, 0]
    y_ticks_positions = np.cumsum([0] + grid_heights[:-1])
    y_tick_labels = names_list
    plt.figure(figsize=(15, 8))
    plt.imshow(combined_grid, cmap=cmap)
    plt.title(f"scaffold_{index}")
    plt.yticks(ticks=y_ticks_positions, labels=y_tick_labels, fontsize=8)
    plt.gca().set_yticks(y_ticks_positions)
    plt.gca().set_yticklabels(y_tick_labels, fontsize=8)
    plt.xticks(ticks=x_ticks_positions, labels=x_ticks_labels)
    plt.draw()
    if path:
        plt.savefig(f'{path}/scaffold_{index}.png', format='png', dpi=500)
    else:
        plt.savefig(f'scaffold_{index}.png', format='png', dpi=500)
    plt.close()
    # plt.show()


# DiemIrisPlot
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
        print(f'Adding ring: {self.rings_added + 1}')
        available_angle = 360 - self.cutout_angle
        angle_scale = available_angle / list_of_thingies[-1][-1]
        color_map = {
            "white": (1.0, 1.0, 1.0),  # White
            "purple": (128 / 255, 0, 128 / 255),  # Purple
            "yellow": (255 / 255, 229 / 255, 0),  # Yellow
            "teal": (0, 128 / 255, 128 / 255),  # Teal
        }
        colors = np.array(list(color_map.values()))
        ring_radius = self.radius - self.rings_added * (self.radius - self.center_radius) / self.number_of_rings

        start_angle_offset = 90
        for index, thing in enumerate(list_of_thingies):
            weights = np.array(thing[0])
            total_weight = np.sum(weights)
            if total_weight == 0:
                blended_rgb = (0, 0, 0)
            else:
                blended_rgb = np.sum(colors.T * weights, axis=1) / total_weight
            blended_hex = mcolors.to_hex(blended_rgb)
            from_angle = start_angle_offset + 360 - (angle_scale * (thing[1] - 1))
            to_angle = start_angle_offset + 360 - (angle_scale * thing[2])
            self.add_wedge(ring_radius, to_angle,from_angle, blended_hex)

        self.rings_added += 1

    def add_heatmap_ring(self, heatmap):
        available_angle = 360 - self.cutout_angle
        angle_scale = available_angle / int(heatmap[-1][-1])
        keys = ["barr", "int", "ovm"]
        values = ["Red", "Blue", "Yellow"]
        color_map = dict(zip(keys, values))

        ring_radius = self.radius + 2 * (self.radius - self.center_radius) / self.number_of_rings

        start_angle_offset = 90
        for index, thing in enumerate(heatmap):
            from_angle = start_angle_offset + 360 - (angle_scale * (int(thing[1]) - 1))
            to_angle = start_angle_offset + 360 - (angle_scale * int(thing[2]))
            self.add_wedge(ring_radius, to_angle, from_angle, color_map[thing[0]])

    def clear_center(self):
        self.add_wedge(self.center_radius, 0, 360, "white")


def diemIrisPlot(input_data, path, names=None, pdf=None, png=None, bed_info=None, length_of_chromosomes=None, heatmap=None, ticks=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_aspect('equal')
    center = (0.5, 0.5)
    radius = 0.5
    number_of_rings = len(input_data)
    cutout_angle = 20
    if heatmap is not None:
        wd = WheelDiagram(ax, center, radius, number_of_rings+1, cutout_angle=cutout_angle)
    else:
        wd = WheelDiagram(ax, center, radius, number_of_rings, cutout_angle=cutout_angle)
    if heatmap is not None:
        wd.add_heatmap_ring(heatmap)
    for ring in input_data:
        wd.add_ring(ring)
    wd.clear_center()

    available_angle = 360 - cutout_angle
    start_angle_offset = 90
    outer_radius = radius + 0.01

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
            angle_deg = np.degrees(midpoint_angle_rad)
            ax.text(label_x, label_y, str(label), ha='center', va='center', fontsize=8, rotation=angle_deg, rotation_mode='anchor')

    # The outer ticks
    if bed_info is not None:
        # Maximum position for scaling
        if heatmap is not None:
            outer_radius_new = radius + 0.035
        else:
            outer_radius_new = outer_radius
        max_position = input_data[0][-1][2]
        for chrom, positions in bed_info.items():
            label_counter = 1  # Initialize a counter
            for label, position in positions:
                angle = start_angle_offset + 360 - (available_angle * position / max_position)
                angle_rad = np.radians(angle)

                x = center[0] + outer_radius_new * np.cos(angle_rad)
                y = center[1] + outer_radius_new * np.sin(angle_rad)

                # if label_counter % 2 == 0:
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

    # Individual names
    if names and len(names) == number_of_rings:
        for i, name in enumerate(names):
            ring_radius = radius - (i + 0.5) * (radius - wd.center_radius) / number_of_rings

            label_angle = 90
            angle_rad = np.radians(label_angle)
            label_x = center[0] + ring_radius * np.cos(angle_rad)
            label_y = center[1] + ring_radius * np.sin(angle_rad)
            ax.text(label_x, label_y, name, ha='right', va='center', fontsize=6, rotation=label_angle - 90,
                    rotation_mode='anchor')

    # Save the output files
    if png:
        plt.savefig(f'{path}/{png}.png', format="png", dpi=500)
    if pdf:
        plt.savefig(f'{path}/{pdf}.pdf', format="pdf", dpi=500)

    if not pdf and not png:
        plt.show()  # Display the plot if desired


# Pairwise distance graph:
class PWC:
    def __init__(self, PWCtallyer, PWCweight, input_path, output_path_results, output_path_heatmap, labels):
        self.PWCtallyer = PWCtallyer
        self.PWCweight = PWCweight
        self.input_path = input_path
        self.output_path_results = output_path_results
        self.output_path_heatmap = output_path_heatmap
        self.U012 = "_012"
        self.labels = labels
        self.PWCtallyer = []
        for i in range(len(self.U012)):
            for j in range(i, len(self.U012)):
                PWCtallyer.append(self.ASJ([self.U012[i], self.U012[j]]))
                if i != j:
                    PWCtallyer.append(self.ASJ([self.U012[j], self.U012[i]]))
        self.PWCweight = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 1, 0]

    @staticmethod
    def SJ(l):
        return "".join(map(str, l))

    @staticmethod
    def ASJ(s):
        return PWC.SJ(s)


    def PWCtally(self, l):
        return [Counter(self.PWCtallyer + l)[k] - 1 for k in self.PWCtallyer]

    def diemDistancePerSite(self, g1, g2):
        pwct = self.PWCtally(["".join(pair) for pair in zip(g1, g2)])
        return np.sum(np.array(pwct) * np.array(self.PWCweight)) / np.sum(pwct[7:])

    @staticmethod
    def CombineRules(ss):
        rules = {
            "00": "0",
            "22": "2",
            "02": "1",
            "20": "1"
        }
        return rules.get(ss, "_")

    @staticmethod
    def diemCombineSites(g1, g2):
        return PWC.ASJ([PWC.CombineRules("".join(pair)) for pair in zip(g1, g2)])

    @staticmethod
    def diemOffspringDistance(p1, p2, o):
        return PWC.diemDistancePerSite(PWC.diemCombineSites(p1, p2), o)

    def pwc_graph(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            RawSyphGenomes = f.read()
        RawSyphGenomes = [s.replace('\n', '') for s in RawSyphGenomes.split("S") if s]
        PWCheatMap = np.zeros((len(RawSyphGenomes), len(RawSyphGenomes)))

        # Calculate pairwise distances
        for i in range(len(RawSyphGenomes)):
            for j in range(len(RawSyphGenomes)):
                PWCheatMap[i, j] = self.diemDistancePerSite(RawSyphGenomes[i], RawSyphGenomes[j])
        np.savetxt(self.output_path_results, PWCheatMap, delimiter=",")
        custom_cmap = LinearSegmentedColormap.from_list("soft_coolwarm", ["#1e90ff", 'white', "#fff266", "#ff1a1a"])
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(PWCheatMap, cmap=custom_cmap, xticklabels=self.labels, yticklabels=self.labels,
                 cbar_kws={"shrink": 1, "fraction": 0.045, "pad": 0.04})
        ax.set_aspect('equal', adjustable='box')
        plt.xticks(rotation=-90)
        plt.savefig(self.output_path_heatmap, format="png", dpi=500)



