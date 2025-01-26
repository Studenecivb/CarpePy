import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from fractions import Fraction

from matplotlib.patches import Wedge
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

from carpePy.diem_helper_functions import *
from .mathematica2python import *


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
    def __init__(self, plot_theme, ind_ids, polarised_data, di_threshold, di_column, phys_res):
        self.polarised_data = polarised_data
        self.di_threshold = di_threshold
        self.di_column = di_column
        self.phys_res = phys_res
        self.plot_theme = plot_theme
        self.ind_ids = ind_ids

        self.diemPlotLabel = None
        self.DIfilteredDATA = None
        self.DIfilteredGenomes = None
        self.DIfilteredHIs = None
        self.DIfilteredBED = None
        self.DIpercent = None
        self.DIfilteredScafRLEs = None
        self.diemDITgenomes = None

        self.diem_plot_prep()

    def diem_plot_prep(self):
        """ Perform DI filtering, dithering, and label generation """
        self.filter_data()

        self.diem_dithering()

        self.generate_plot_label(self.plot_theme)

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
    else:
        Kb = sgl
        Kb = np.array(Kb).astype(int)
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


def diemUnitPlot(chromosome_data, index, names_list, bed_data, path=None):
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
    x_ticks = chr_kb_ticks(bed_data).astype(int)
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


def diemIrisPlot(input_data, path, names=None, pdf=None, png=None, bed_info=None, length_of_chromosomes=None, heatmap=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_aspect('equal')
    center = (.5, .5)
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

    # Chromosome names
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
        plt.savefig(f'{path}/{png}.png', format="png")
    if pdf:
        plt.savefig(f'{path}/{pdf}.pdf', format="pdf")

    if not pdf and not png:
        plt.show()  # Display the plot if desired


