package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zlahqr is an auxiliary routine called by CHSEQR to update the
//    eigenvalues and Schur decomposition already computed by CHSEQR, by
//    dealing with the Hessenberg submatrix in rows and columns ILO to
//    IHI.
func Zlahqr(wantt, wantz bool, n, ilo, ihi int, h *mat.CMatrix, w *mat.CVector, iloz, ihiz int, z *mat.CMatrix) (info int) {
	var h11, h11s, h22, one, sc, sum, t, t1, temp, u, v2, x, y, zero complex128
	var aa, ab, ba, bb, dat1, h10, h21, half, rone, rtemp, rzero, s, safmax, safmin, smlnum, sx, t2, tst, ulp float64
	var i, i1, i2, itmax, its, j, jhi, jlo, k, l, m, nh, nz int

	v := cvf(2)

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rzero = 0.0
	rone = 1.0
	half = 0.5
	dat1 = 3.0 / 4.0

	//     Quick return if possible
	if n == 0 {
		return
	}
	if ilo == ihi {
		w.Set(ilo-1, h.Get(ilo-1, ilo-1))
		return
	}

	//     ==== clear out the trash ====
	for j = ilo; j <= ihi-3; j++ {
		h.Set(j+2-1, j-1, zero)
		h.Set(j+3-1, j-1, zero)
	}
	if ilo <= ihi-2 {
		h.Set(ihi-1, ihi-2-1, zero)
	}
	//     ==== ensure that subdiagonal entries are real ====
	if wantt {
		jlo = 1
		jhi = n
	} else {
		jlo = ilo
		jhi = ihi
	}
	for i = ilo + 1; i <= ihi; i++ {
		if h.GetIm(i-1, i-1-1) != rzero {
			//           ==== The following redundant normalization
			//           .    avoids problems with both gradual and
			//           .    sudden underflow in ABS(H(I,I-1)) ====
			sc = h.Get(i-1, i-1-1) / complex(cabs1(h.Get(i-1, i-1-1)), 0)
			sc = cmplx.Conj(sc) / complex(cmplx.Abs(sc), 0)
			h.SetRe(i-1, i-1-1, h.GetMag(i-1, i-1-1))
			h.Off(i-1, i-1).CVector().Scal(jhi-i+1, sc, h.Rows)
			h.Off(jlo-1, i-1).CVector().Scal(min(jhi, i+1)-jlo+1, cmplx.Conj(sc), 1)
			if wantz {
				z.Off(iloz-1, i-1).CVector().Scal(ihiz-iloz+1, cmplx.Conj(sc), 1)
			}
		}
	}

	nh = ihi - ilo + 1
	nz = ihiz - iloz + 1

	//     Set machine-dependent constants for the stopping criterion.
	safmin = Dlamch(SafeMinimum)
	safmax = rone / safmin
	safmin, safmax = Dlabad(safmin, safmax)
	ulp = Dlamch(Precision)
	smlnum = safmin * (float64(nh) / ulp)

	//     I1 and I2 are the indices of the first row and last column of H
	//     to which transformations must be applied. If eigenvalues only are
	//     being computed, I1 and I2 are set inside the main loop.
	if wantt {
		i1 = 1
		i2 = n
	}

	//     ITMAX is the total number of QR iterations allowed.
	itmax = 30 * max(10, nh)

	//     The main loop begins here. I is the loop index and decreases from
	//     IHI to ILO in steps of 1. Each iteration of the loop works
	//     with the active submatrix in rows and columns L to I.
	//     Eigenvalues I+1 to IHI have already converged. Either L = ILO, or
	//     H(L,L-1) is negligible so that the matrix splits.
	i = ihi
label30:
	;
	if i < ilo {
		return
	}

	//     Perform QR iterations on rows and columns ILO to I until a
	//     submatrix of order 1 splits off at the bottom because a
	//     subdiagonal element has become negligible.
	l = ilo
	for its = 0; its <= itmax; its++ {
		//        Look for a single small subdiagonal element.
		for k = i; k >= l+1; k-- {
			if cabs1(h.Get(k-1, k-1-1)) <= smlnum {
				goto label50
			}
			tst = cabs1(h.Get(k-1-1, k-1-1)) + cabs1(h.Get(k-1, k-1))
			if complex(tst, 0) == zero {
				if k-2 >= ilo {
					tst = tst + math.Abs(h.GetRe(k-1-1, k-2-1))
				}
				if k+1 <= ihi {
					tst = tst + math.Abs(h.GetRe(k, k-1))
				}
			}
			//           ==== The following is a conservative small subdiagonal
			//           .    deflation criterion due to Ahues & Tisseur (LAWN 122,
			//           .    1997). It has better mathematical foundation and
			//           .    improves accuracy in some examples.  ====
			if math.Abs(h.GetRe(k-1, k-1-1)) <= ulp*tst {
				ab = math.Max(cabs1(h.Get(k-1, k-1-1)), cabs1(h.Get(k-1-1, k-1)))
				ba = math.Min(cabs1(h.Get(k-1, k-1-1)), cabs1(h.Get(k-1-1, k-1)))
				aa = math.Max(cabs1(h.Get(k-1, k-1)), cabs1(h.Get(k-1-1, k-1-1)-h.Get(k-1, k-1)))
				bb = math.Min(cabs1(h.Get(k-1, k-1)), cabs1(h.Get(k-1-1, k-1-1)-h.Get(k-1, k-1)))
				s = aa + ab
				if ba*(ab/s) <= math.Max(smlnum, ulp*(bb*(aa/s))) {
					goto label50
				}
			}
		}
	label50:
		;
		l = k
		if l > ilo {
			//           H(L,L-1) is negligible
			h.Set(l-1, l-1-1, zero)
		}

		//        Exit from loop if a submatrix of order 1 has split off.
		if l >= i {
			goto label140
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
			s = dat1 * math.Abs(h.GetRe(l, l-1))
			t = complex(s, 0) + h.Get(l-1, l-1)
		} else if its == 20 {
			//           Exceptional shift.
			s = dat1 * math.Abs(h.GetRe(i-1, i-1-1))
			t = complex(s, 0) + h.Get(i-1, i-1)
		} else {
			//           Wilkinson's shift.
			t = h.Get(i-1, i-1)
			u = cmplx.Sqrt(h.Get(i-1-1, i-1)) * cmplx.Sqrt(h.Get(i-1, i-1-1))
			s = cabs1(u)
			if s != rzero {
				x = complex(half, 0) * (h.Get(i-1-1, i-1-1) - t)
				sx = cabs1(x)
				s = math.Max(s, cabs1(x))
				y = complex(s, 0) * cmplx.Sqrt(cmplx.Pow(x/complex(s, 0), 2)+cmplx.Pow(u/complex(s, 0), 2))
				if sx > rzero {
					if real(x/complex(sx, 0))*real(y)+imag(x/complex(sx, 0))*imag(y) < rzero {
						y = -y
					}
				}
				t = t - u*Zladiv(u, x+y)
			}
		}

		//        Look for two consecutive small subdiagonal elements.
		for m = i - 1; m >= l+1; m-- { //
			//           Determine the effect of starting the single-shift QR
			//           iteration at row M, and see if this would make H(M,M-1)
			//           negligible.
			h11 = h.Get(m-1, m-1)
			h22 = h.Get(m, m)
			h11s = h11 - t
			h21 = h.GetRe(m, m-1)
			s = cabs1(h11s) + math.Abs(h21)
			h11s = h11s / complex(s, 0)
			h21 = h21 / s
			v.Set(0, h11s)
			v.SetRe(1, h21)
			h10 = h.GetRe(m-1, m-1-1)
			if math.Abs(h10)*math.Abs(h21) <= ulp*(cabs1(h11s)*(cabs1(h11)+cabs1(h22))) {
				goto label70
			}
		}
		h11 = h.Get(l-1, l-1)
		h22 = h.Get(l, l)
		h11s = h11 - t
		h21 = h.GetRe(l, l-1)
		s = cabs1(h11s) + math.Abs(h21)
		h11s = h11s / complex(s, 0)
		h21 = h21 / s
		v.Set(0, h11s)
		v.SetRe(1, h21)
	label70:
		;

		//        Single-shift QR step
		for k = m; k <= i-1; k++ {
			//           The first iteration of this loop determines a reflection G
			//           from the vector V and applies it from left and right to H,
			//           thus creating a nonzero bulge below the subdiagonal.
			//
			//           Each subsequent iteration determines a reflection G to
			//           restore the Hessenberg form in the (K-1)th column, and thus
			//           chases the bulge one step toward the bottom of the active
			//           submatrix.
			//
			//           V(2) is always real before the call to ZLARFG, and hence
			//           after the call T2 ( = T1*V(2) ) is also real.
			if k > m {
				v.Copy(2, h.Off(k-1, k-1-1).CVector(), 1, 1)
			}
			*v.GetPtr(0), t1 = Zlarfg(2, v.Get(0), v.Off(1), 1)
			if k > m {
				h.Set(k-1, k-1-1, v.Get(0))
				h.Set(k, k-1-1, zero)
			}
			v2 = v.Get(1)
			t2 = real(t1 * v2)

			//           Apply G from the left to transform the rows of the matrix
			//           in columns K to I2.
			for j = k; j <= i2; j++ {
				sum = cmplx.Conj(t1)*h.Get(k-1, j-1) + complex(t2, 0)*h.Get(k, j-1)
				h.Set(k-1, j-1, h.Get(k-1, j-1)-sum)
				h.Set(k, j-1, h.Get(k, j-1)-sum*v2)
			}

			//           Apply G from the right to transform the columns of the
			//           matrix in rows I1 to min(K+2,I).
			for j = i1; j <= min(k+2, i); j++ {
				sum = t1*h.Get(j-1, k-1) + complex(t2, 0)*h.Get(j-1, k)
				h.Set(j-1, k-1, h.Get(j-1, k-1)-sum)
				h.Set(j-1, k, h.Get(j-1, k)-sum*cmplx.Conj(v2))
			}

			if wantz {
				//              Accumulate transformations in the matrix Z
				for j = iloz; j <= ihiz; j++ {
					sum = t1*z.Get(j-1, k-1) + complex(t2, 0)*z.Get(j-1, k)
					z.Set(j-1, k-1, z.Get(j-1, k-1)-sum)
					z.Set(j-1, k, z.Get(j-1, k)-sum*cmplx.Conj(v2))
				}
			}

			if k == m && m > l {
				//              If the QR step was started at row M > L because two
				//              consecutive small subdiagonals were found, then extra
				//              scaling must be performed to ensure that H(M,M-1) remains
				//              real.
				temp = one - t1
				temp = temp / complex(cmplx.Abs(temp), 0)
				h.Set(m, m-1, h.Get(m, m-1)*cmplx.Conj(temp))
				if m+2 <= i {
					h.Set(m+2-1, m, h.Get(m+2-1, m)*temp)
				}
				for j = m; j <= i; j++ {
					if j != m+1 {
						if i2 > j {
							h.Off(j-1, j).CVector().Scal(i2-j, temp, h.Rows)
						}
						h.Off(i1-1, j-1).CVector().Scal(j-i1, cmplx.Conj(temp), 1)
						if wantz {
							z.Off(iloz-1, j-1).CVector().Scal(nz, cmplx.Conj(temp), 1)
						}
					}
				}
			}
		}

		//        Ensure that H(I,I-1) is real.
		temp = h.Get(i-1, i-1-1)
		if imag(temp) != rzero {
			rtemp = cmplx.Abs(temp)
			h.SetRe(i-1, i-1-1, rtemp)
			temp = temp / complex(rtemp, 0)
			if i2 > i {
				h.Off(i-1, i).CVector().Scal(i2-i, cmplx.Conj(temp), h.Rows)
			}
			h.Off(i1-1, i-1).CVector().Scal(i-i1, temp, 1)
			if wantz {
				z.Off(iloz-1, i-1).CVector().Scal(nz, temp, 1)
			}
		}

	}

	//     Failure to converge in remaining number of iterations
	info = i
	return

label140:
	;

	//     H(I,I-1) is negligible: one eigenvalue has converged.
	w.Set(i-1, h.Get(i-1, i-1))

	//     return to start of the main loop with new value of I.
	i = l - 1
	goto label30
}
