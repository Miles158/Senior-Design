import numpy as np
from scipy.special import erfc




def _gray(bit_position):
    """Make adjacent points in the constellation diagram differ by only one digit."""
    label = bit_position ^ (bit_position >> 1)
    return label

def normalize(points):
    """Scale a constellation"""
    points = np.asarray(points, dtype=complex)
    return points / np.sqrt(np.mean(np.abs(points) ** 2))

def psk(num_points):
    """Gray-labeled M-PSK (M = 2, 4, 8, ...)."""
    points = np.zeros(num_points, dtype=complex)
    spacing = 360 / num_points                     
    offset = spacing / 2 if num_points > 2 else 0     

    for position in range(num_points):
        angle = np.radians(position * spacing + offset)
        location = np.cos(angle) + 1j * np.sin(angle)  
        label = _gray(position)                       
        points[label] = location

    return points


def qam(num_points):
    """Gray-labeled square M-QAM (M = 4, 16, 64, 256, ...)."""
    sqrt_pts = int(round(np.sqrt(num_points)))
    bit_per_axis = int(np.log2(sqrt_pts))  
    levels = np.arange(-sqrt_pts + 1, sqrt_pts, 2)
    points = np.zeros(num_points, dtype=complex)

    for hort_idx in range(sqrt_pts):
        for vert_idx in range(sqrt_pts):
            location = levels[hort_idx] + 1j * levels[vert_idx] # location of point in complex plane
            hort_label = _gray(hort_idx)  # bits for the column
            vert_label = _gray(vert_idx)  # bits for the row
            label = hort_label * sqrt_pts + vert_label             # column bits first, then row bits
            points[label] = location

    return normalize(points)


def bits_per_symbol(const): 
    num_of_bits = int(np.log2(len(const)))
    return num_of_bits


def modulate(bits, const): #
    """Given a constellation, map bits to symbols"""

    bit_per_symbol = bits_per_symbol(const)
    vectors_of_bits = bits.reshape(-1, bit_per_symbol)     
    labels = np.zeros(len(vectors_of_bits), dtype=int)
    for column in range(bit_per_symbol):
        labels = labels * 2 + vectors_of_bits[:, column]
    labels = []
    for bit_group in vectors_of_bits:      # one symbol at a time
        label = 0
        for bit in bit_group:             
            label = label * 2 + bit
        labels.append(label)
    labels = np.array(labels)

    return np.asarray(const)[labels]


def demodulate(rx, const):
    k = bits_per_symbol(const)

    best_label = np.zeros(len(rx), dtype=int)
    best_dist = np.full(len(rx), np.inf)
    for label, p in enumerate(const):
        dist = np.abs(rx - p) ** 2
        closer = dist < best_dist
        best_label[closer] = label
        best_dist[closer] = dist[closer]
    return ((best_label[:, None] >> np.arange(k)[::-1]) & 1).ravel()



CONSTELLATIONS = {
    "bpsk": psk(2),
    "qpsk": psk(4),
    "8psk": psk(8),
    "16qam": qam(16),
}



def encode(bits, scheme):
    return modulate(bits, CONSTELLATIONS[scheme])
