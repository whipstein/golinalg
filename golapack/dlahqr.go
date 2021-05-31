package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlahqr is an auxiliary routine called by DHSEQR to update the
//    eigenvalues and Schur decomposition already computed by DHSEQR, by
//    dealing with the Hessenberg submatrix in rows and columns ILO to
//    IHI.
func Dlahqr(wantt, wantz bool, n, ilo, ihi *int, h *mat.Matrix, ldh *int, wr, wi *mat.Vector, iloz, ihiz *int, z *mat.Matrix, ldz, info *int) {
	var aa, ab, ba, bb, cs, dat1, dat2, det, h11, h12, h21, h21s, h22, one, rt1i, rt1r, rt2i, rt2r, rtdisc, s, safmax, safmin, smlnum, sn, sum, t1, t2, t3, tr, tst, two, ulp, v2, v3, zero float64
	var i, i1, i2, itmax, its, j, k, l, m, nh, nr, nz int

	v := vf(3)

	zero = 0.0
	one = 1.0
	two = 2.0
	dat1 = 3.0 / 4.0
	dat2 = -0.4375

	(*info) = 0

	//     Quick return if possible
	if (*n) == 0 {
		return
	}
	if (*ilo) == (*ihi) {
		wr.Set((*ilo)-1, h.Get((*ilo)-1, (*ilo)-1))
		wi.Set((*ilo)-1, zero)
		return
	}

	//     ==== clear out the trash ====
	for j = (*ilo); j <= (*ihi)-3; j++ {
		h.Set(j+2-1, j-1, zero)
		h.Set(j+3-1, j-1, zero)
	}
	if (*ilo) <= (*ihi)-2 {
		h.Set((*ihi)-1, (*ihi)-2-1, zero)
	}

	nh = (*ihi) - (*ilo) + 1
	nz = (*ihiz) - (*iloz) + 1

	//     Set machine-dependent constants for the stopping criterion.
	safmin = Dlamch(SafeMinimum)
	safmax = one / safmin
	Dlabad(&safmin, &safmax)
	ulp = Dlamch(Precision)
	smlnum = safmin * (float64(nh) / ulp)

	//     I1 and I2 are the indices of the first row and last column of H
	//     to which transformations must be applied. If eigenvalues only are
	//     being computed, I1 and I2 are set inside the main loop.
	if wantt {
		i1 = 1
		i2 = (*n)
	}

	//     ITMAX is the total number of QR iterations allowed.
	itmax = 30 * maxint(10, nh)

	//     The main loop begins here. I is the loop index and decreases from
	//     IHI to ILO in steps of 1 or 2. Each iteration of the loop works
	//     with the active submatrix in rows and columns L to I.
	//     Eigenvalues I+1 to IHI have already converged. Either L = ILO or
	//     H(L,L-1) is negligible so that the matrix splits.
	i = (*ihi)
label20:
	;
	l = (*ilo)
	if i < (*ilo) {
		return
	}

	//     Perform QR iterations on rows and columns ILO to I until a
	//     submatrix of order 1 or 2 splits off at the bottom because a
	//     subdiagonal element has become negligible.
	for its = 0; its <= itmax; its++ {
		//        Look for a single small subdiagonal element.
		for k = i; k >= l+1; k-- {
			if math.Abs(h.Get(k-1, k-1-1)) <= smlnum {
				goto label40
			}
			tst = math.Abs(h.Get(k-1-1, k-1-1)) + math.Abs(h.Get(k-1, k-1))
			if tst == zero {
				if k-2 >= (*ilo) {
					tst = tst + math.Abs(h.Get(k-1-1, k-2-1))
				}
				if k+1 <= (*ihi) {
					tst = tst + math.Abs(h.Get(k+1-1, k-1))
				}
			}
			//           ==== The following is a conservative small subdiagonal
			//           .    deflation  criterion due to Ahues & Tisseur (LAWN 122,
			//           .    1997). It has better mathematical foundation and
			//           .    improves accuracy in some cases.  ====
			if math.Abs(h.Get(k-1, k-1-1)) <= ulp*tst {
				ab = maxf64(math.Abs(h.Get(k-1, k-1-1)), math.Abs(h.Get(k-1-1, k-1)))
				ba = minf64(math.Abs(h.Get(k-1, k-1-1)), math.Abs(h.Get(k-1-1, k-1)))
				aa = maxf64(math.Abs(h.Get(k-1, k-1)), math.Abs(h.Get(k-1-1, k-1-1)-h.Get(k-1, k-1)))
				bb = minf64(math.Abs(h.Get(k-1, k-1)), math.Abs(h.Get(k-1-1, k-1-1)-h.Get(k-1, k-1)))
				s = aa + ab
				if ba*(ab/s) <= maxf64(smlnum, ulp*(bb*(aa/s))) {
					goto label40
				}
			}
		}
	label40:
		;
		l = k
		if l > (*ilo) {
			//           H(L,L-1) is negligible
			h.Set(l-1, l-1-1, zero)
		}

		//        Exit from loop if a submatrix of order 1 or 2 has split off.
		if l >= i-1 {
			goto label150
		}

		//        Now the active submatrix is in rows and columns L to I. If
		//        eigenvalues only are being computed, only the active submatrix
		//        need be transformed.
		if !wantt {
			i1 = l
			i2 = i
		}

		if its == 10 {
			//           Exceptional shift.
			s = math.Abs(h.Get(l+1-1, l-1)) + math.Abs(h.Get(l+2-1, l+1-1))
			h11 = dat1*s + h.Get(l-1, l-1)
			h12 = dat2 * s
			h21 = s
			h22 = h11
		} else if its == 20 {
			//           Exceptional shift.
			s = math.Abs(h.Get(i-1, i-1-1)) + math.Abs(h.Get(i-1-1, i-2-1))
			h11 = dat1*s + h.Get(i-1, i-1)
			h12 = dat2 * s
			h21 = s
			h22 = h11
		} else {
			//           Prepare to use Francis' double shift
			//           (i.e. 2nd degree generalized Rayleigh quotient)
			h11 = h.Get(i-1-1, i-1-1)
			h21 = h.Get(i-1, i-1-1)
			h12 = h.Get(i-1-1, i-1)
			h22 = h.Get(i-1, i-1)
		}
		s = math.Abs(h11) + math.Abs(h12) + math.Abs(h21) + math.Abs(h22)
		if s == zero {
			rt1r = zero
			rt1i = zero
			rt2r = zero
			rt2i = zero
		} else {
			h11 = h11 / s
			h21 = h21 / s
			h12 = h12 / s
			h22 = h22 / s
			tr = (h11 + h22) / two
			det = (h11-tr)*(h22-tr) - h12*h21
			rtdisc = math.Sqrt(math.Abs(det))
			if det >= zero {
				//              ==== complex conjugate shifts ====
				rt1r = tr * s
				rt2r = rt1r
				rt1i = rtdisc * s
				rt2i = -rt1i
			} else {
				//              ==== real shifts (use only one of them)  ====
				rt1r = tr + rtdisc
				rt2r = tr - rtdisc
				if math.Abs(rt1r-h22) <= math.Abs(rt2r-h22) {
					rt1r = rt1r * s
					rt2r = rt1r
				} else {
					rt2r = rt2r * s
					rt1r = rt2r
				}
				rt1i = zero
				rt2i = zero
			}
		}
		//        Look for two consecutive small subdiagonal elements.
		for m = i - 2; m >= l; m-- {
			//           Determine the effect of starting the double-shift QR
			//           iteration at row M, and see if this would make H(M,M-1)
			//           negligible.  (The following uses scaling to avoid
			//           overflows and most underflows.)
			h21s = h.Get(m+1-1, m-1)
			s = math.Abs(h.Get(m-1, m-1)-rt2r) + math.Abs(rt2i) + math.Abs(h21s)
			h21s = h.Get(m+1-1, m-1) / s
			v.Set(0, h21s*h.Get(m-1, m+1-1)+(h.Get(m-1, m-1)-rt1r)*((h.Get(m-1, m-1)-rt2r)/s)-rt1i*(rt2i/s))
			v.Set(1, h21s*(h.Get(m-1, m-1)+h.Get(m+1-1, m+1-1)-rt1r-rt2r))
			v.Set(2, h21s*h.Get(m+2-1, m+1-1))
			s = math.Abs(v.Get(0)) + math.Abs(v.Get(1)) + math.Abs(v.Get(2))
			v.Set(0, v.Get(0)/s)
			v.Set(1, v.Get(1)/s)
			v.Set(2, v.Get(2)/s)
			if m == l {
				goto label60
			}
			if math.Abs(h.Get(m-1, m-1-1))*(math.Abs(v.Get(1))+math.Abs(v.Get(2))) <= ulp*math.Abs(v.Get(0))*(math.Abs(h.Get(m-1-1, m-1-1))+math.Abs(h.Get(m-1, m-1))+math.Abs(h.Get(m+1-1, m+1-1))) {
				goto label60
			}
		}
	label60:
		;

		//        Double-shift QR step
		for k = m; k <= i-1; k++ {
			//           The first iteration of this loop determines a reflection G
			//           from the vector V and applies it from left and right to H,
			//           thus creating a nonzero bulge below the subdiagonal.
			//
			//           Each subsequent iteration determines a reflection G to
			//           restore the Hessenberg form in the (K-1)th column, and thus
			//           chases the bulge one step toward the bottom of the active
			//           submatrix. NR is the order of G.
			nr = minint(3, i-k+1)
			if k > m {
				goblas.Dcopy(&nr, h.Vector(k-1, k-1-1), toPtr(1), v, toPtr(1))
			}
			Dlarfg(&nr, v.GetPtr(0), v.Off(1), func() *int { y := 1; return &y }(), &t1)
			if k > m {
				h.Set(k-1, k-1-1, v.Get(0))
				h.Set(k+1-1, k-1-1, zero)
				if k < i-1 {
					h.Set(k+2-1, k-1-1, zero)
				}
			} else if m > l {
				//               ==== Use the following instead of
				//               .    H( K, K-1 ) = -H( K, K-1 ) to
				//               .    avoid a bug when v(2) and v(3)
				//               .    underflow. ====
				h.Set(k-1, k-1-1, h.Get(k-1, k-1-1)*(one-t1))
			}
			v2 = v.Get(1)
			t2 = t1 * v2
			if nr == 3 {
				v3 = v.Get(2)
				t3 = t1 * v3

				//              Apply G from the left to transform the rows of the matrix
				//              in columns K to I2.
				for j = k; j <= i2; j++ {
					sum = h.Get(k-1, j-1) + v2*h.Get(k+1-1, j-1) + v3*h.Get(k+2-1, j-1)
					h.Set(k-1, j-1, h.Get(k-1, j-1)-sum*t1)
					h.Set(k+1-1, j-1, h.Get(k+1-1, j-1)-sum*t2)
					h.Set(k+2-1, j-1, h.Get(k+2-1, j-1)-sum*t3)
				}

				//              Apply G from the right to transform the columns of the
				//              matrix in rows I1 to minf64(K+3,I).
				for j = i1; j <= minint(k+3, i); j++ {
					sum = h.Get(j-1, k-1) + v2*h.Get(j-1, k+1-1) + v3*h.Get(j-1, k+2-1)
					h.Set(j-1, k-1, h.Get(j-1, k-1)-sum*t1)
					h.Set(j-1, k+1-1, h.Get(j-1, k+1-1)-sum*t2)
					h.Set(j-1, k+2-1, h.Get(j-1, k+2-1)-sum*t3)
				}

				if wantz {
					//                 Accumulate transformations in the matrix Z
					for j = (*iloz); j <= (*ihiz); j++ {
						sum = z.Get(j-1, k-1) + v2*z.Get(j-1, k+1-1) + v3*z.Get(j-1, k+2-1)
						z.Set(j-1, k-1, z.Get(j-1, k-1)-sum*t1)
						z.Set(j-1, k+1-1, z.Get(j-1, k+1-1)-sum*t2)
						z.Set(j-1, k+2-1, z.Get(j-1, k+2-1)-sum*t3)
					}
				}
			} else if nr == 2 {

				//              Apply G from the left to transform the rows of the matrix
				//              in columns K to I2.
				for j = k; j <= i2; j++ {
					sum = h.Get(k-1, j-1) + v2*h.Get(k+1-1, j-1)
					h.Set(k-1, j-1, h.Get(k-1, j-1)-sum*t1)
					h.Set(k+1-1, j-1, h.Get(k+1-1, j-1)-sum*t2)
				}

				//              Apply G from the right to transform the columns of the
				//              matrix in rows I1 to minf64(K+3,I).
				for j = i1; j <= i; j++ {
					sum = h.Get(j-1, k-1) + v2*h.Get(j-1, k+1-1)
					h.Set(j-1, k-1, h.Get(j-1, k-1)-sum*t1)
					h.Set(j-1, k+1-1, h.Get(j-1, k+1-1)-sum*t2)
				}

				if wantz {
					//                 Accumulate transformations in the matrix Z
					for j = (*iloz); j <= (*ihiz); j++ {
						sum = z.Get(j-1, k-1) + v2*z.Get(j-1, k+1-1)
						z.Set(j-1, k-1, z.Get(j-1, k-1)-sum*t1)
						z.Set(j-1, k+1-1, z.Get(j-1, k+1-1)-sum*t2)
					}
				}
			}
		}

	}

	//     Failure to converge in remaining number of iterations
	(*info) = i
	return

label150:
	;

	if l == i {
		//        H(I,I-1) is negligible: one eigenvalue has converged.
		wr.Set(i-1, h.Get(i-1, i-1))
		wi.Set(i-1, zero)
	} else if l == i-1 {
		//        H(I-1,I-2) is negligible: a pair of eigenvalues have converged.
		//
		//        Transform the 2-by-2 submatrix to standard Schur form,
		//        and compute and store the eigenvalues.
		Dlanv2(h.GetPtr(i-1-1, i-1-1), h.GetPtr(i-1-1, i-1), h.GetPtr(i-1, i-1-1), h.GetPtr(i-1, i-1), wr.GetPtr(i-1-1), wi.GetPtr(i-1-1), wr.GetPtr(i-1), wi.GetPtr(i-1), &cs, &sn)

		if wantt {
			//           Apply the transformation to the rest of H.
			if i2 > i {
				goblas.Drot(toPtr(i2-i), h.Vector(i-1-1, i+1-1), ldh, h.Vector(i-1, i+1-1), ldh, &cs, &sn)
			}
			goblas.Drot(toPtr(i-i1-1), h.Vector(i1-1, i-1-1), toPtr(1), h.Vector(i1-1, i-1), toPtr(1), &cs, &sn)
		}
		if wantz {
			//           Apply the transformation to Z.
			goblas.Drot(&nz, z.Vector((*iloz)-1, i-1-1), toPtr(1), z.Vector((*iloz)-1, i-1), toPtr(1), &cs, &sn)
		}
	}

	//     return to start of the main loop with new value of I.
	i = l - 1
	goto label20
}
