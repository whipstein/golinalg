package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Dlatm4 generates basic square matrices, which may later be
// multiplied by others in order to produce test matrices.  It is
// intended mainly to be used to test the generalized eigenvalue
// routines.
//
// It first generates the diagonal and (possibly) subdiagonal,
// according to the value of ITYPE, NZ1, NZ2, ISIGN, AMAGN, and RCOND.
// It then fills in the upper triangle with random numbers, if TRIANG is
// non-zero.
func Dlatm4(itype, n, nz1, nz2, isign *int, amagn, rcond, triang *float64, idist *int, iseed *[]int, a *mat.Matrix, lda *int) {
	var alpha, cl, cr, half, one, safmin, sl, sr, sv1, sv2, temp, two, zero float64
	var i, ioff, isdb, isde, jc, jd, jr, k, kbeg, kend, klen int

	zero = 0.0
	one = 1.0
	two = 2.0
	half = one / two

	if (*n) <= 0 {
		return
	}
	golapack.Dlaset('F', n, n, &zero, &zero, a, lda)

	//     Insure a correct ISEED
	if ((*iseed)[3] % 2) != 1 {
		(*iseed)[3] = (*iseed)[3] + 1
	}

	//     Compute diagonal and subdiagonal according to ITYPE, NZ1, NZ2,
	//     and RCOND
	if (*itype) != 0 {
		if absint(*itype) >= 4 {
			kbeg = maxint(1, minint(*n, (*nz1)+1))
			kend = maxint(kbeg, minint(*n, (*n)-(*nz2)))
			klen = kend + 1 - kbeg
		} else {
			kbeg = 1
			kend = (*n)
			klen = (*n)
		}
		isdb = 1
		isde = 0
		switch absint(*itype) {
		case 1:
			goto label10
		case 2:
			goto label30
		case 3:
			goto label50
		case 4:
			goto label80
		case 5:
			goto label100
		case 6:
			goto label120
		case 7:
			goto label140
		case 8:
			goto label160
		case 9:
			goto label180
		case 10:
			goto label200
		}

		//        abs(ITYPE) = 1: Identity
	label10:
		;
		for jd = 1; jd <= (*n); jd++ {
			a.Set(jd-1, jd-1, one)
		}
		goto label220

		//        abs(ITYPE) = 2: Transposed Jordan block
	label30:
		;
		for jd = 1; jd <= (*n)-1; jd++ {
			a.Set(jd+1-1, jd-1, one)
		}
		isdb = 1
		isde = (*n) - 1
		goto label220

		//        abs(ITYPE) = 3: Transposed Jordan block, followed by the
		//                        identity.
	label50:
		;
		k = ((*n) - 1) / 2
		for jd = 1; jd <= k; jd++ {
			a.Set(jd+1-1, jd-1, one)
		}
		isdb = 1
		isde = k
		for jd = k + 2; jd <= 2*k+1; jd++ {
			a.Set(jd-1, jd-1, one)
		}
		goto label220

		//        abs(ITYPE) = 4: 1,...,k
	label80:
		;
		for jd = kbeg; jd <= kend; jd++ {
			a.Set(jd-1, jd-1, float64(jd-(*nz1)))
		}
		goto label220

		//        abs(ITYPE) = 5: One large D value:
	label100:
		;
		for jd = kbeg + 1; jd <= kend; jd++ {
			a.Set(jd-1, jd-1, (*rcond))
		}
		a.Set(kbeg-1, kbeg-1, one)
		goto label220

		//        abs(ITYPE) = 6: One small D value:
	label120:
		;
		for jd = kbeg; jd <= kend-1; jd++ {
			a.Set(jd-1, jd-1, one)
		}
		a.Set(kend-1, kend-1, (*rcond))
		goto label220

		//        abs(ITYPE) = 7: Exponentially distributed D values:
	label140:
		;
		a.Set(kbeg-1, kbeg-1, one)
		if klen > 1 {
			alpha = math.Pow(*rcond, one/float64(klen-1))
			for i = 2; i <= klen; i++ {
				a.Set((*nz1)+i-1, (*nz1)+i-1, math.Pow(alpha, float64(i-1)))
			}
		}
		goto label220

		//        abs(ITYPE) = 8: Arithmetically distributed D values:
	label160:
		;
		a.Set(kbeg-1, kbeg-1, one)
		if klen > 1 {
			alpha = (one - (*rcond)) / float64(klen-1)
			for i = 2; i <= klen; i++ {
				a.Set((*nz1)+i-1, (*nz1)+i-1, float64(klen-i)*alpha+(*rcond))
			}
		}
		goto label220

		//        abs(ITYPE) = 9: Randomly distributed D values on ( RCOND, 1):
	label180:
		;
		alpha = math.Log(*rcond)
		for jd = kbeg; jd <= kend; jd++ {
			a.Set(jd-1, jd-1, math.Exp(alpha*matgen.Dlaran(iseed)))
		}
		goto label220

		//        abs(ITYPE) = 10: Randomly distributed D values from DIST
	label200:
		;
		for jd = kbeg; jd <= kend; jd++ {
			a.Set(jd-1, jd-1, matgen.Dlarnd(idist, iseed))
		}

	label220:
		;

		//        Scale by AMAGN
		for jd = kbeg; jd <= kend; jd++ {
			a.Set(jd-1, jd-1, (*amagn)*float64(a.Get(jd-1, jd-1)))
		}
		for jd = isdb; jd <= isde; jd++ {
			a.Set(jd+1-1, jd-1, (*amagn)*float64(a.Get(jd+1-1, jd-1)))
		}

		//        If ISIGN = 1 or 2, assign random signs to diagonal and
		//        subdiagonal
		if (*isign) > 0 {
			for jd = kbeg; jd <= kend; jd++ {
				if float64(a.Get(jd-1, jd-1)) != zero {
					if matgen.Dlaran(iseed) > half {
						a.Set(jd-1, jd-1, -a.Get(jd-1, jd-1))
					}
				}
			}
			for jd = isdb; jd <= isde; jd++ {
				if float64(a.Get(jd+1-1, jd-1)) != zero {
					if matgen.Dlaran(iseed) > half {
						a.Set(jd+1-1, jd-1, -a.Get(jd+1-1, jd-1))
					}
				}
			}
		}

		//        Reverse if ITYPE < 0
		if (*itype) < 0 {
			for jd = kbeg; jd <= (kbeg+kend-1)/2; jd++ {
				temp = a.Get(jd-1, jd-1)
				a.Set(jd-1, jd-1, a.Get(kbeg+kend-jd-1, kbeg+kend-jd-1))
				a.Set(kbeg+kend-jd-1, kbeg+kend-jd-1, temp)
			}
			for jd = 1; jd <= ((*n)-1)/2; jd++ {
				temp = a.Get(jd+1-1, jd-1)
				a.Set(jd+1-1, jd-1, a.Get((*n)+1-jd-1, (*n)-jd-1))
				a.Set((*n)+1-jd-1, (*n)-jd-1, temp)
			}
		}

		//        If ISIGN = 2, and no subdiagonals already, then apply
		//        random rotations to make 2x2 blocks.
		if (*isign) == 2 && (*itype) != 2 && (*itype) != 3 {
			safmin = golapack.Dlamch(SafeMinimum)
			for jd = kbeg; jd <= kend-1; jd += 2 {
				if matgen.Dlaran(iseed) > half {
					//                 Rotation on left.
					cl = two*matgen.Dlaran(iseed) - one
					sl = two*matgen.Dlaran(iseed) - one
					temp = one / maxf64(safmin, math.Sqrt(math.Pow(cl, 2)+math.Pow(sl, 2)))
					cl = cl * temp
					sl = sl * temp

					//                 Rotation on right.
					cr = two*matgen.Dlaran(iseed) - one
					sr = two*matgen.Dlaran(iseed) - one
					temp = one / maxf64(safmin, math.Sqrt(math.Pow(cr, 2)+math.Pow(sr, 2)))
					cr = cr * temp
					sr = sr * temp

					//                 Apply
					sv1 = a.Get(jd-1, jd-1)
					sv2 = a.Get(jd+1-1, jd+1-1)
					a.Set(jd-1, jd-1, cl*cr*sv1+sl*sr*sv2)
					a.Set(jd+1-1, jd-1, -sl*cr*sv1+cl*sr*sv2)
					a.Set(jd-1, jd+1-1, -cl*sr*sv1+sl*cr*sv2)
					a.Set(jd+1-1, jd+1-1, sl*sr*sv1+cl*cr*sv2)
				}
			}
		}

	}

	//     Fill in upper triangle (except for 2x2 blocks)
	if (*triang) != zero {
		if (*isign) != 2 || (*itype) == 2 || (*itype) == 3 {
			ioff = 1
		} else {
			ioff = 2
			for jr = 1; jr <= (*n)-1; jr++ {
				if a.Get(jr+1-1, jr-1) == zero {
					a.Set(jr-1, jr+1-1, (*triang)*matgen.Dlarnd(idist, iseed))
				}
			}
		}

		for jc = 2; jc <= (*n); jc++ {
			for jr = 1; jr <= jc-ioff; jr++ {
				a.Set(jr-1, jc-1, (*triang)*matgen.Dlarnd(idist, iseed))
			}
		}
	}
}
