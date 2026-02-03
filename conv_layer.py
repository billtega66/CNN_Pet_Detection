
import random

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        # Initialize filters with random values
        self.filters = [
            [
                [random.gauss(0, 1) for _ in range(3)]
                for _ in range(3)
            ]
            for _ in range(num_filters)
        ]

    def sliding_window(self,img):
        """Generates all 3x3 image patches using a sliding window."""
        heights, widths = len(img), len(img[0])
        for height in range(heights - 2):
            for width in range(widths - 2):
                patch = [
                    [img[height + m][width + n] for n in range(3)]
                    for m in range(3)
                ]
                yield patch, height, width

    def forward_pass(self, input):
        """Applies the convolution operation to the input image."""
        heights, widths = len(input), len(input[0])
        output = [
            [
                [0 for _ in range(widths - 2)]
                for _ in range(heights - 2)
            ]
            for _ in range(self.num_filters)
        ]

        for patch, h, w in self.sliding_window(input):
            for f in range(self.num_filters):
                conv_sum = 0
                for i in range(3):
                    for j in range(3):
                        conv_sum += patch[i][j] * self.filters[f][i][j]
                output[f][h][w] = conv_sum

        return output