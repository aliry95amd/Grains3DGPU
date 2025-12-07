#bin/bash

python3 generate_xml.py --L 0.576 --H 0.576 --n 128 --r 0.02 --type S1 --output S1_N1
python3 generate_xml.py --L 0.576 --H 0.576 --n 128 --r 0.02 --type S4 --output S4_N1
python3 generate_xml.py --L 0.576 --H 0.576 --n 128 --r 0.02 --type B1 --output B1_N1
python3 generate_xml.py --L 0.576 --H 0.576 --n 128 --r 0.02 --type B4 --output B4_N1

python3 generate_xml.py --L 1.152 --H 1.152 --n 4096 --r 0.02 --type S1 --output S1_N2
python3 generate_xml.py --L 1.152 --H 1.152 --n 4096 --r 0.02 --type S4 --output S4_N2
python3 generate_xml.py --L 1.152 --H 1.152 --n 4096 --r 0.02 --type B1 --output B1_N2
python3 generate_xml.py --L 1.152 --H 1.152 --n 4096 --r 0.02 --type B4 --output B4_N2

python3 generate_xml.py --L 2.304 --H 2.304 --n 32768 --r 0.02 --type S1 --output S1_N3
python3 generate_xml.py --L 2.304 --H 2.304 --n 32768 --r 0.02 --type S4 --output S4_N3
python3 generate_xml.py --L 2.304 --H 2.304 --n 32768 --r 0.02 --type B1 --output B1_N3
python3 generate_xml.py --L 2.304 --H 2.304 --n 32768 --r 0.02 --type B4 --output B4_N3

python3 generate_xml.py --L 4.608 --H 4.608 --n 534288 --r 0.02 --type S1 --output S1_N4
python3 generate_xml.py --L 4.608 --H 4.608 --n 534288 --r 0.02 --type S4 --output S4_N4
python3 generate_xml.py --L 4.608 --H 4.608 --n 534288 --r 0.02 --type B1 --output B1_N4
python3 generate_xml.py --L 4.608 --H 4.608 --n 534288 --r 0.02 --type B4 --output B4_N4