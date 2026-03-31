# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

"""Multi-treatment univariate encoding example code.

Executing this code will generate a model from the data file and print various
statistics.
"""

import pandas
import kuplift
import logging
from pprint import pprint  # So that the user does not have to import it.

logger = logging.getLogger(__name__)


def main():
    global ue  # Make it available for study after execution of this function.
    logging.basicConfig(level=logging.DEBUG)
    ue = kuplift.MultiTreatmentUnivariateEncoding()


if __name__ == "__main__":
    main()