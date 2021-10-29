package golapack

import "math"

// Dlasv2 computes the singular value decomposition of a 2-by-2
// triangular matrix
//    [  F   G  ]
//    [  0   H  ].
// On return, math.Abs(SSMAX) is the larger singular value, math.Abs(SSMIN) is the
// smaller singular value, and (CSL,SNL) and (CSR,SNR) are the left and
// right singular vectors for math.Abs(SSMAX), giving the decomposition
//
//    [ CSL  SNL ] [  F   G  ] [ CSR -SNR ]  =  [ SSMAX   0   ]
//    [-SNL  CSL ] [  0   H  ] [ SNR  CSR ]     [  0    SSMIN ].
func Dlasv2(f, g, h float64) (ssmin, ssmax, snr, csr, snl, csl float64) {
	var gasmal, swap bool
	var a, clt, crt, d, fa, four, ft, ga, gt, ha, half, ht, l, m, mm, one, r, s, slt, srt, t, temp, tsign, tt, two, zero float64
	var pmax int

	zero = 0.0
	half = 0.5
	one = 1.0
	two = 2.0
	four = 4.0

	ft = f
	fa = math.Abs(ft)
	ht = h
	ha = math.Abs(h)

	//     PMAX points to the maximum absolute element of matrix
	//       PMAX = 1 if F largest in absolute values
	//       PMAX = 2 if G largest in absolute values
	//       PMAX = 3 if H largest in absolute values
	pmax = 1
	swap = (ha > fa)
	if swap {
		pmax = 3
		temp = ft
		ft = ht
		ht = temp
		temp = fa
		fa = ha
		ha = temp
		//
		//        Now FA .ge. HA
		//
	}
	gt = g
	ga = math.Abs(gt)
	if ga == zero {
		//
		//        Diagonal matrix
		//
		ssmin = ha
		ssmax = fa
		clt = one
		crt = one
		slt = zero
		srt = zero
	} else {
		gasmal = true
		if ga > fa {
			pmax = 2
			if (fa / ga) < Dlamch(Epsilon) {
				//              Case of very large GA
				gasmal = false
				ssmax = ga
				if ha > one {
					ssmin = fa / (ga / ha)
				} else {
					ssmin = (fa / ga) * ha
				}
				clt = one
				slt = ht / gt
				srt = one
				crt = ft / gt
			}
		}
		if gasmal {
			//           Normal case
			d = fa - ha
			if d == fa {
				//              Copes with infinite F or H
				l = one
			} else {
				l = d / fa
			}

			//           Note that 0 .le. L .le. 1
			m = gt / ft

			//           Note that math.Abs(M) .le. 1/macheps
			t = two - l

			//           Note that T .ge. 1
			mm = m * m
			tt = t * t
			s = math.Sqrt(tt + mm)

			//           Note that 1 .le. S .le. 1 + 1/macheps
			if l == zero {
				r = math.Abs(m)
			} else {
				r = math.Sqrt(l*l + mm)
			}

			//           Note that 0 .le. R .le. 1 + 1/macheps
			a = half * (s + r)

			//           Note that 1 .le. A .le. 1 + math.Abs(M)
			ssmin = ha / a
			ssmax = fa * a
			if mm == zero {
				//              Note that M is very tiny
				if l == zero {
					t = math.Copysign(two, ft) * math.Copysign(one, gt)
				} else {
					t = gt/math.Copysign(d, ft) + m/t
				}
			} else {
				t = (m/(s+t) + m/(r+l)) * (one + a)
			}
			l = math.Sqrt(t*t + four)
			crt = two / l
			srt = t / l
			clt = (crt + srt*m) / a
			slt = (ht / ft) * srt / a
		}
	}
	if swap {
		csl = srt
		snl = crt
		csr = slt
		snr = clt
	} else {
		csl = clt
		snl = slt
		csr = crt
		snr = srt
	}

	//     Correct signs of SSMAX and SSMIN
	if pmax == 1 {
		tsign = math.Copysign(one, csr) * math.Copysign(one, csl) * math.Copysign(one, f)
	}
	if pmax == 2 {
		tsign = math.Copysign(one, snr) * math.Copysign(one, csl) * math.Copysign(one, g)
	}
	if pmax == 3 {
		tsign = math.Copysign(one, snr) * math.Copysign(one, snl) * math.Copysign(one, h)
	}
	ssmax = math.Copysign(ssmax, tsign)
	ssmin = math.Copysign(ssmin, tsign*math.Copysign(one, f)*math.Copysign(one, h))

	return
}
