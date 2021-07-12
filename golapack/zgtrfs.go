package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgtrfs improves the computed solution to a system of linear
// equations when the coefficient matrix is tridiagonal, and provides
// error bounds and backward error estimates for the solution.
func Zgtrfs(trans byte, n, nrhs *int, dl, d, du, dlf, df, duf, du2 *mat.CVector, ipiv *[]int, b *mat.CMatrix, ldb *int, x *mat.CMatrix, ldx *int, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector, info *int) {
	var notran bool
	var transn, transt byte
	var eps, lstres, one, s, safe1, safe2, safmin, three, two, zero float64
	var count, i, itmax, j, kase, nz int
	isave := make([]int, 3)

	itmax = 5
	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0

	Cabs1 := func(zdum complex128) float64 { return math.Abs(real(zdum)) + math.Abs(imag(zdum)) }

	//     Test the input parameters.
	(*info) = 0
	notran = trans == 'N'
	if !notran && trans != 'T' && trans != 'C' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*ldb) < max(1, *n) {
		(*info) = -13
	} else if (*ldx) < max(1, *n) {
		(*info) = -15
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGTRFS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		for j = 1; j <= (*nrhs); j++ {
			ferr.Set(j-1, zero)
			berr.Set(j-1, zero)
		}
		return
	}

	if notran {
		transn = 'N'
		transt = 'C'
	} else {
		transn = 'C'
		transt = 'N'
	}

	//     NZ = maximum number of nonzero elements in each row of A, plus 1
	nz = 4
	eps = Dlamch(Epsilon)
	safmin = Dlamch(SafeMinimum)
	safe1 = float64(nz) * safmin
	safe2 = safe1 / eps

	//     Do for each right hand side
	for j = 1; j <= (*nrhs); j++ {

		count = 1
		lstres = three
	label20:
		;

		//        Loop until stopping criterion is satisfied.
		//
		//        Compute residual R = B - op(A) * X,
		//        where op(A) = A, A**T, or A**H, depending on TRANS.
		goblas.Zcopy(*n, b.CVector(0, j-1, 1), work.Off(0, 1))
		Zlagtm(trans, n, func() *int { y := 1; return &y }(), toPtrf64(-one), dl, d, du, x.Off(0, j-1), ldx, &one, work.CMatrix(*n, opts), n)

		//        Compute abs(op(A))*abs(x) + abs(b) for use in the backward
		//        error bound.
		if notran {
			if (*n) == 1 {
				rwork.Set(0, Cabs1(b.Get(0, j-1))+Cabs1(d.Get(0))*Cabs1(x.Get(0, j-1)))
			} else {
				rwork.Set(0, Cabs1(b.Get(0, j-1))+Cabs1(d.Get(0))*Cabs1(x.Get(0, j-1))+Cabs1(du.Get(0))*Cabs1(x.Get(1, j-1)))
				for i = 2; i <= (*n)-1; i++ {
					rwork.Set(i-1, Cabs1(b.Get(i-1, j-1))+Cabs1(dl.Get(i-1-1))*Cabs1(x.Get(i-1-1, j-1))+Cabs1(d.Get(i-1))*Cabs1(x.Get(i-1, j-1))+Cabs1(du.Get(i-1))*Cabs1(x.Get(i, j-1)))
				}
				rwork.Set((*n)-1, Cabs1(b.Get((*n)-1, j-1))+Cabs1(dl.Get((*n)-1-1))*Cabs1(x.Get((*n)-1-1, j-1))+Cabs1(d.Get((*n)-1))*Cabs1(x.Get((*n)-1, j-1)))
			}
		} else {
			if (*n) == 1 {
				rwork.Set(0, Cabs1(b.Get(0, j-1))+Cabs1(d.Get(0))*Cabs1(x.Get(0, j-1)))
			} else {
				rwork.Set(0, Cabs1(b.Get(0, j-1))+Cabs1(d.Get(0))*Cabs1(x.Get(0, j-1))+Cabs1(dl.Get(0))*Cabs1(x.Get(1, j-1)))
				for i = 2; i <= (*n)-1; i++ {
					rwork.Set(i-1, Cabs1(b.Get(i-1, j-1))+Cabs1(du.Get(i-1-1))*Cabs1(x.Get(i-1-1, j-1))+Cabs1(d.Get(i-1))*Cabs1(x.Get(i-1, j-1))+Cabs1(dl.Get(i-1))*Cabs1(x.Get(i, j-1)))
				}
				rwork.Set((*n)-1, Cabs1(b.Get((*n)-1, j-1))+Cabs1(du.Get((*n)-1-1))*Cabs1(x.Get((*n)-1-1, j-1))+Cabs1(d.Get((*n)-1))*Cabs1(x.Get((*n)-1, j-1)))
			}
		}

		//        Compute componentwise relative backward error from formula
		//
		//        max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
		//
		//        where abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
		s = zero
		for i = 1; i <= (*n); i++ {
			if rwork.Get(i-1) > safe2 {
				s = math.Max(s, Cabs1(work.Get(i-1))/rwork.Get(i-1))
			} else {
				s = math.Max(s, (Cabs1(work.Get(i-1))+safe1)/(rwork.Get(i-1)+safe1))
			}
		}
		berr.Set(j-1, s)

		//        Test stopping criterion. Continue iterating if
		//           1) The residual BERR(J) is larger than machine epsilon, and
		//           2) BERR(J) decreased by at least a factor of 2 during the
		//              last iteration, and
		//           3) At most ITMAX iterations tried.
		if berr.Get(j-1) > eps && two*berr.Get(j-1) <= lstres && count <= itmax {
			//           Update solution and try again.
			Zgttrs(trans, n, func() *int { y := 1; return &y }(), dlf, df, duf, du2, ipiv, work.CMatrix(*n, opts), n, info)
			goblas.Zaxpy(*n, complex(one, 0), work.Off(0, 1), x.CVector(0, j-1, 1))
			lstres = berr.Get(j - 1)
			count = count + 1
			goto label20
		}

		//        Bound error from formula
		//
		//        norm(X - XTRUE) / norm(X) .le. FERR =
		//        norm( abs(inv(op(A)))*
		//           ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X)
		//
		//        where
		//          norm(Z) is the magnitude of the largest component of Z
		//          inv(op(A)) is the inverse of op(A)
		//          abs(Z) is the componentwise absolute value of the matrix or
		//             vector Z
		//          NZ is the maximum number of nonzeros in any row of A, plus 1
		//          EPS is machine epsilon
		//
		//        The i-th component of abs(R)+NZ*EPS*(abs(op(A))*abs(X)+abs(B))
		//        is incremented by SAFE1 if the i-th component of
		//        abs(op(A))*abs(X) + abs(B) is less than SAFE2.
		//
		//        Use ZLACN2 to estimate the infinity-norm of the matrix
		//           inv(op(A)) * diag(W),
		//        where W = abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) )))
		for i = 1; i <= (*n); i++ {
			if rwork.Get(i-1) > safe2 {
				rwork.Set(i-1, Cabs1(work.Get(i-1))+float64(nz)*eps*rwork.Get(i-1))
			} else {
				rwork.Set(i-1, Cabs1(work.Get(i-1))+float64(nz)*eps*rwork.Get(i-1)+safe1)
			}
		}

		kase = 0
	label70:
		;
		Zlacn2(n, work.Off((*n)), work, ferr.GetPtr(j-1), &kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Multiply by diag(W)*inv(op(A)**H).
				Zgttrs(transt, n, func() *int { y := 1; return &y }(), dlf, df, duf, du2, ipiv, work.CMatrix(*n, opts), n, info)
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, rwork.GetCmplx(i-1)*work.Get(i-1))
				}
			} else {
				//              Multiply by inv(op(A))*diag(W).
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, rwork.GetCmplx(i-1)*work.Get(i-1))
				}
				Zgttrs(transn, n, func() *int { y := 1; return &y }(), dlf, df, duf, du2, ipiv, work.CMatrix(*n, opts), n, info)
			}
			goto label70
		}

		//        Normalize error.
		lstres = zero
		for i = 1; i <= (*n); i++ {
			lstres = math.Max(lstres, Cabs1(x.Get(i-1, j-1)))
		}
		if lstres != zero {
			ferr.Set(j-1, ferr.Get(j-1)/lstres)
		}

	}
}
