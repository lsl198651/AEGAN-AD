# m-n(machine-noise)_<machine_index>
# f-n(factory-noise)_<factory_noise_index>
# n-lv(noise-level)_<noise_level_index>
# 可以把收缩舒张期类型做进去
CLASS_ATTRI_ALL = {'fan': ['m-n', 'f-n', 'n-lv'],
                   'gearbox': ['volt', 'wt', 'id'],
                   'bearing': ['vel', 'loc', 'f-n'],
                   'slider': ['vel', 'ac', 'f-n'],
                   'ToyCar': ['car', 'spd', 'mic', 'noise'],
                   'ToyTrain': ['car', 'spd', 'mic', 'noise'],
                   'valve': ['pat', 'panel', 'v'],
                   'heart': ['Systolic', 'Diastolic']}

CLASS_SEC_ATTRI = {'fan': [['m-n'], ['f-n'], ['n-lv'], ['m-n'], ['f-n'], ['n-lv']],
                   'gearbox': [['volt'], ['wt'], ['id'], ['volt'], ['wt'], ['id']],
                   'bearing': [['vel'], ['vel', 'loc'], ['vel', 'f-n'], ['vel'], ['vel', 'loc'], ['vel', 'f-n']],
                   'slider': [['vel'], ['ac'], ['f-n'], ['vel'], ['ac'], ['f-n']],
                   'ToyCar': [['car', 'spd', 'mic', 'noise'], ['car', 'spd', 'mic', 'noise'], ['car', 'spd', 'mic', 'noise'], ['car', 'spd', 'mic', 'noise'], ['car', 'spd', 'mic', 'noise'], ['car', 'spd', 'mic', 'noise']],
                   'ToyTrain': [['car', 'spd', 'mic', 'noise'], ['car', 'spd', 'mic', 'noise'], ['car', 'spd', 'mic', 'noise'], ['car', 'spd', 'mic', 'noise'], ['car', 'spd', 'mic', 'noise'], ['car', 'spd', 'mic', 'noise']],
                   'valve': [['pat'], ['pat', 'panel'], ['v'], ['pat'], ['pat', 'panel'], ['v']],
                   'heart': [['Systolic'], ['Diastolic'], ['Systolic'], ['Diastolic']]}

ATTRI_CODE = {'fan': {'m-n': {'none': 0, 'W': 1, 'X': 2, 'Y': 3, 'Z': 4},
                      'f-n': {'none': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4},
                      'n-lv': {'none': 0, 'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4}},
              'gearbox': {'volt': {'none': 0, '0.6': 1, '0.8': 2, '1.0': 3, '1.2': 4, '1.3': 5, '1.5': 6, '1.7': 7, '1.8': 8, '2.0': 9, '2.2': 10, '2.3': 11, '2.5': 12, '2.7': 13, '2.8': 14, '3.0': 15, '3.3': 16, '3.5': 17},
                          'wt': {'none': 0, '0': 1, '20': 2, '30': 3, '50': 4, '70': 5, '80': 6, '100': 7, '120': 8, '130': 9, '150': 10, '170': 11, '180': 12, '200': 13, '230': 14, '250': 15},
                          'id': {'none': 0, '00': 1, '01': 2, '02': 3, '03': 4, '04': 5, '05': 6, '06': 7, '07': 8, '08': 9, '11': 10, '12': 11, '13': 12}},
              'bearing': {'vel': {'none': 0, '2': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8, '11': 9, '12': 10, '13': 11, '14': 12, '15': 13, '16': 14, '17': 15, '18': 16, '19': 17, '20': 18, '21': 19, '22': 20, '24': 21, '26': 22},
                          'loc': {'none': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8},
                          'f-n': {'none': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}},
              'slider': {'vel': {'none': 0, '100': 1, '140': 2, '175': 3, '200': 4, '210': 5, '250': 6, '290': 7, '300': 8, '325': 9, '360': 10, '400': 11, '500': 12, '600': 13, '700': 14, '800': 15, '900': 16, '1000': 17, '1100': 18, '1200': 19, '1300': 20},
                         'ac': {'none': 0, '0.01': 1, '0.02': 2, '0.03': 3, '0.04': 4, '0.05': 5, '0.06': 6, '0.07': 7, '0.08': 8, '0.09': 9, '0.10': 10, '0.11': 11, '0.12': 12, '0.14': 13},
                         'f-n': {'none': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}},
              'ToyCar': {'car': {'none': 0, 'A1': 1, 'A2': 2, 'C1': 3, 'C2': 4, 'E1': 5, 'E2': 6, 'F1': 7, 'F2': 8, 'G1': 9, 'G2': 10, 'H1': 11, 'H2': 12},
                         'spd': {'none': 0, '28V': 1, '31V': 2, '34V': 3, '37V': 4, '40V': 5},
                         'mic': {'none': 0, '1': 1, '2': 2},
                         'noise': {'none': 0, '1': 1, '2': 2, '6': 3}},
              'ToyTrain': {'car': {'none': 0, 'A1': 1, 'A2': 2, 'C1': 3, 'C2': 4, 'E1': 5, 'E2': 6, 'F1': 7, 'F2': 8, 'G1': 9, 'G2': 10, 'H1': 11, 'H2': 12},
                           'spd': {'none': 0, '6': 1, '7': 2, '8': 3, '9': 4, '10': 5},
                           'mic': {'none': 0, '1': 1, '2': 2},
                           'noise': {'none': 0, '1': 1, '2': 2, '6': 3, '7': 4}},
              'valve': {'pat': {'none': 0, '00': 1, '01': 2, '02': 3, '03': 4, '04': 5, '05': 6, '06': 7, '07': 8, '04_05': 9, '05_05': 10},
                        'panel': {'none': 0, 'open': 1, 'b-c': 2, 's-c': 3, 'bs-c': 4},
                        'v': {'none': 0, 'v1pat': 1, 'v2pat': 2, 'v1pat_v2pat': 3}},
              'heart': {'Systolic': {'none': 0, 'early': 1, 'holo': 2, 'mid': 3, 'late': 4},
                        'Diastolic': {'none': 0, 'early': 1}}}

HEART_ATTRI_ALL = {['Systolic', 'Diastolic']}
HEARRT_SEC_ATTRI = {[['Systolic'], ['Diastolic'], ['Systolic'], ['Diastolic']]}
HEART_ATTRI_CODE = {'Systolic': {'early': 0, 'holo': 1, 'mid': 2, 'late': 3, 'none': 4},
                    'Diastolic': {'early': 0, 'none': 1}}
