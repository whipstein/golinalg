package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zstein computes the eigenvectors of a real symmetric tridiagonal
// matrix T corresponding to specified eigenvalues, using inverse
// iteration.
//
// The maximum number of iterations allowed for each eigenvector is
// specified by an internal parameter MAXITS (currently set to 5).
//
// Although the eigenvectors are real, they are stored in a complex
// array, which may be passed to ZUNMTR or ZUPMTR for back
// transformation to the eigenvectors of a complex Hermitian matrix
// which was reduced to tridiagonal form.
func Zstein(n *int, d, e *mat.Vector, m *int, w *mat.Vector, iblock, isplit *[]int, z *mat.CMatrix, ldz *int, work *mat.Vector, iwork, ifail *[]int, info *int) {
	var cone, czero complex128
	var dtpcrt, eps, eps1, nrm, odm1, odm3, one, onenrm, ortol, pertol, scl, sep, ten, tol, xj, xjm, zero, ztr float64
	var b1, blksiz, bn, extra, gpind, i, iinfo, indrv1, indrv2, indrv3, indrv4, indrv5, its, j, j1, jblk, jmax, jr, maxits, nblk, nrmchk int
	iseed := make([]int, 4)

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0
	ten = 1.0e+1
	odm3 = 1.0e-3
	odm1 = 1.0e-1
	maxits = 5
	extra = 2

	//     Test the input parameters.
	(*info) = 0
	for i = 1; i <= (*m); i++ {
		(*ifail)[i-1] = 0
	}

	if (*n) < 0 {
		(*info) = -1
	} else if (*m) < 0 || (*m) > (*n) {
		(*info) = -4
	} else if (*ldz) < maxint(1, *n) {
		(*info) = -9
	} else {
		for j = 2; j <= (*m); j++ {
			if (*iblock)[j-1] < (*iblock)[j-1-1] {
				(*info) = -6
				goto label30
			}
			if (*iblock)[j-1] == (*iblock)[j-1-1] && w.Get(j-1) < w.Get(j-1-1) {
				(*info) = -5
				goto label30
			}
		}
	label30:
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSTEIN"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*m) == 0 {
		return
	} else if (*n) == 1 {
		z.Set(0, 0, cone)
		return
	}

	//     Get machine constants.
	eps = Dlamch(Precision)

	//     Initialize seed for random number generator DLARNV.
	for i = 1; i <= 4; i++ {
		iseed[i-1] = 1
	}

	//     Initialize pointers.
	indrv1 = 0
	indrv2 = indrv1 + (*n)
	indrv3 = indrv2 + (*n)
	indrv4 = indrv3 + (*n)
	indrv5 = indrv4 + (*n)

	//     Compute eigenvectors of matrix blocks.
	j1 = 1
	for nblk = 1; nblk <= (*iblock)[(*m)-1]; nblk++ {
		//        Find starting and ending indices of block nblk.
		if nblk == 1 {
			b1 = 1
		} else {
			b1 = (*isplit)[nblk-1-1] + 1
		}
		bn = (*isplit)[nblk-1]
		blksiz = bn - b1 + 1
		if blksiz == 1 {
			goto label60
		}
		gpind = j1

		//        Compute reorthogonalization criterion and stopping criterion.
		onenrm = d.GetMag(b1-1) + e.GetMag(b1-1)
		onenrm = maxf64(onenrm, d.GetMag(bn-1)+e.GetMag(bn-1-1))
		for i = b1 + 1; i <= bn-1; i++ {
			onenrm = maxf64(onenrm, d.GetMag(i-1)+e.GetMag(i-1-1)+e.GetMag(i-1))
		}
		ortol = odm3 * onenrm

		dtpcrt = math.Sqrt(odm1 / float64(blksiz))

		//        Loop through eigenvalues of block nblk.
	label60:
		;
		jblk = 0
		for j = j1; j <= (*m); j++ {
			if (*iblock)[j-1] != nblk {
				j1 = j
				goto label180
			}
			jblk = jblk + 1
			xj = w.Get(j - 1)

			//           Skip all the work if the block size is one.
			if blksiz == 1 {
				work.Set(indrv1+1-1, one)
				goto label140
			}

			//           If eigenvalues j and j-1 are too close, add a relatively
			//           small perturbation.
			if jblk > 1 {
				eps1 = math.Abs(eps * xj)
				pertol = ten * eps1
				sep = xj - xjm
				if sep < pertol {
					xj = xjm + pertol
				}
			}

			its = 0
			nrmchk = 0

			//           Get random starting vector.
			Dlarnv(func() *int { y := 2; return &y }(), &iseed, &blksiz, work.Off(indrv1+1-1))

			//           Copy the matrix T so it won't be destroyed in factorization.
			goblas.Dcopy(blksiz, d.Off(b1-1), 1, work.Off(indrv4+1-1), 1)
			goblas.Dcopy(blksiz-1, e.Off(b1-1), 1, work.Off(indrv2+2-1), 1)
			goblas.Dcopy(blksiz-1, e.Off(b1-1), 1, work.Off(indrv3+1-1), 1)

			//           Compute LU factors with partial pivoting  ( PT = LU )
			tol = zero
			Dlagtf(&blksiz, work.Off(indrv4+1-1), &xj, work.Off(indrv2+2-1), work.Off(indrv3+1-1), &tol, work.Off(indrv5+1-1), iwork, &iinfo)

			//           Update iteration count.
		label70:
			;
			its = its + 1
			if its > maxits {
				goto label120
			}

			//           Normalize and scale the righthand side vector Pb.
			jmax = goblas.Idamax(blksiz, work.Off(indrv1+1-1), 1)
			scl = float64(blksiz) * onenrm * maxf64(eps, work.GetMag(indrv4+blksiz-1)) / work.GetMag(indrv1+jmax-1)
			goblas.Dscal(blksiz, scl, work.Off(indrv1+1-1), 1)

			//           Solve the system LU = Pb.
			Dlagts(toPtr(-1), &blksiz, work.Off(indrv4+1-1), work.Off(indrv2+2-1), work.Off(indrv3+1-1), work.Off(indrv5+1-1), iwork, work.Off(indrv1+1-1), &tol, &iinfo)

			//           Reorthogonalize by modified Gram-Schmidt if eigenvalues are
			//           close enough.
			if jblk == 1 {
				goto label110
			}
			if math.Abs(xj-xjm) > ortol {
				gpind = j
			}
			if gpind != j {
				for i = gpind; i <= j-1; i++ {
					ztr = zero
					for jr = 1; jr <= blksiz; jr++ {
						ztr = ztr + work.Get(indrv1+jr-1)*z.GetRe(b1-1+jr-1, i-1)
					}
					for jr = 1; jr <= blksiz; jr++ {
						work.Set(indrv1+jr-1, work.Get(indrv1+jr-1)-ztr*z.GetRe(b1-1+jr-1, i-1))
					}
				}
			}

			//           Check the infinity norm of the iterate.
		label110:
			;
			jmax = goblas.Idamax(blksiz, work.Off(indrv1+1-1), 1)
			nrm = work.GetMag(indrv1 + jmax - 1)

			//           Continue for additional iterations after norm reaches
			//           stopping criterion.
			if nrm < dtpcrt {
				goto label70
			}
			nrmchk = nrmchk + 1
			if nrmchk < extra+1 {
				goto label70
			}

			goto label130

			//           If stopping criterion was not satisfied, update info and
			//           store eigenvector number in array ifail.
		label120:
			;
			(*info) = (*info) + 1
			(*ifail)[(*info)-1] = j

			//           Accept iterate as jth eigenvector.
		label130:
			;
			scl = one / goblas.Dnrm2(blksiz, work.Off(indrv1+1-1), 1)
			jmax = goblas.Idamax(blksiz, work.Off(indrv1+1-1), 1)
			if work.Get(indrv1+jmax-1) < zero {
				scl = -scl
			}
			goblas.Dscal(blksiz, scl, work.Off(indrv1+1-1), 1)
		label140:
			;
			for i = 1; i <= (*n); i++ {
				z.Set(i-1, j-1, czero)
			}
			for i = 1; i <= blksiz; i++ {
				z.SetRe(b1+i-1-1, j-1, work.Get(indrv1+i-1))
			}

			//           Save the shift to check eigenvalue spacing at next
			//           iteration.
			xjm = xj

		}
	label180:
	}
}
