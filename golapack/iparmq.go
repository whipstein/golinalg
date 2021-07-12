package golapack

import "math"

// Iparmq This program sets problem and machine dependent parameters
//      useful for xHSEQR and related subroutines for eigenvalue
//      problems. It is called whenever
//      IPARMQ is called with 12 <= ISPEC <= 16
func Iparmq(ispec *int, name, opts []byte, n, ilo, ihi, lwork *int) (iparmqReturn int) {
	var two float64
	var i, iacc22, ic, inibl, inmin, inwin, ishfts, iz, k22min, kacmin, knwswp, nh, nibble, nmin, ns int

	inmin = 12
	inwin = 13
	inibl = 14
	ishfts = 15
	iacc22 = 16
	nmin = 75
	k22min = 14
	kacmin = 14
	nibble = 14
	knwswp = 500
	two = 2.0

	if ((*ispec) == ishfts) || ((*ispec) == inwin) || ((*ispec) == iacc22) {
		//        ==== Set the number simultaneous shifts ====
		nh = (*ihi) - (*ilo) + 1
		ns = 2
		if nh >= 30 {
			ns = 4
		}
		if nh >= 60 {
			ns = 10
		}
		if nh >= 150 {
			ns = max(10, nh/int(math.Round(math.Log(float64(nh))/math.Log(two))))
		}
		if nh >= 590 {
			ns = 64
		}
		if nh >= 3000 {
			ns = 128
		}
		if nh >= 6000 {
			ns = 256
		}
		ns = max(2, ns-ns%2)
	}

	if (*ispec) == inmin {
		//        ===== Matrices of order smaller than NMIN get sent
		//        .     to xLAHQR, the classic double shift algorithm.
		//        .     This must be at least 11. ====
		iparmqReturn = nmin

	} else if (*ispec) == inibl {
		//        ==== INIBL: skip a multi-shift qr iteration and
		//        .    whenever aggressive early deflation finds
		//        .    at least (NIBBLE*(window size)/100) deflations. ====
		iparmqReturn = nibble

	} else if (*ispec) == ishfts {
		//        ==== NSHFTS: The number of simultaneous shifts =====
		iparmqReturn = ns

	} else if (*ispec) == inwin {
		//        ==== NW: deflation window size.  ====
		if nh <= knwswp {
			iparmqReturn = ns
		} else {
			iparmqReturn = 3 * ns / 2
		}

	} else if (*ispec) == iacc22 {
		//        ==== IACC22: Whether to accumulate reflections
		//        .     before updating the far-from-diagonal elements
		//        .     and whether to use 2-by-2 block structure while
		//        .     doing it.  A small amount of work could be saved
		//        .     by making this choice dependent also upon the
		//        .     NH=IHI-ILO+1.
		//
		//
		//        Convert NAME to upper case if the first character is lower case.
		iparmqReturn = 0
		subnam := []byte(name)
		ic = int(subnam[0])
		iz = int('Z')
		if iz == 90 || iz == 122 {
			//           ASCII character set
			if ic >= 97 && ic <= 122 {
				subnam[0] = byte(ic - 32)
				for i = 2; i <= 6; i++ {
					ic = int(subnam[i-1])
					if ic >= 97 && ic <= 122 {
						subnam[i-1] = byte(ic - 32)
					}
				}
			}

		} else if iz == 233 || iz == 169 {
			//           EBCDIC character set
			if (ic >= 129 && ic <= 137) || (ic >= 145 && ic <= 153) || (ic >= 162 && ic <= 169) {
				subnam[0] = byte(ic + 64)
				for i = 2; i <= 6; i++ {
					ic = int(subnam[i-1])
					if (ic >= 129 && ic <= 137) || (ic >= 145 && ic <= 153) || (ic >= 162 && ic <= 169) {
						subnam[i-1] = byte(ic + 64)
					}
				}
			}

		} else if iz == 218 || iz == 250 {
			//           Prime machines:  ASCII+128
			if ic >= 225 && ic <= 250 {
				subnam[0] = byte(ic - 32)
				for i = 2; i <= 6; i++ {
					ic = int(subnam[i-1])
					if ic >= 225 && ic <= 250 {
						subnam[i-1] = byte(ic - 32)
					}
				}
			}
		}

		if string(subnam[1:6]) == "GGHRD" || string(subnam[1:6]) == "GGHD3" {
			iparmqReturn = 1
			if nh >= k22min {
				iparmqReturn = 2
			}
		} else if string(subnam[3:6]) == "EXC" {
			if nh >= kacmin {
				iparmqReturn = 1
			}
			if nh >= k22min {
				iparmqReturn = 2
			}
		} else if string(subnam[1:6]) == "HSEQR" || string(subnam[1:5]) == "LAQR" {
			if ns >= kacmin {
				iparmqReturn = 1
			}
			if ns >= k22min {
				iparmqReturn = 2
			}
		}

	} else {
		//        ===== invalid value of ispec =====
		iparmqReturn = -1

	}
	return
}
