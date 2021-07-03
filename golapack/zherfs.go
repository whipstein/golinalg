package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zherfs improves the computed solution to a system of linear
// equations when the coefficient matrix is Hermitian indefinite, and
// provides error bounds and backward error estimates for the solution.
func Zherfs(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, af *mat.CMatrix, ldaf *int, ipiv *[]int, b *mat.CMatrix, ldb *int, x *mat.CMatrix, ldx *int, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector, info *int) {
	var upper bool
	var one complex128
	var eps, lstres, s, safe1, safe2, safmin, three, two, xk, zero float64
	var count, i, itmax, j, k, kase, nz int
	var err error
	_ = err

	isave := make([]int, 3)

	itmax = 5
	zero = 0.0
	one = (1.0 + 0.0*1i)
	two = 2.0
	three = 3.0

	Cabs1 := func(zdum complex128) float64 { return math.Abs(real(zdum)) + math.Abs(imag(zdum)) }

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldaf) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -10
	} else if (*ldx) < maxint(1, *n) {
		(*info) = -12
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHERFS"), -(*info))
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

	//     NZ = maximum number of nonzero elements in each row of A, plus 1
	nz = (*n) + 1
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
		//        Compute residual R = B - A * X
		goblas.Zcopy(*n, b.CVector(0, j-1), 1, work, 1)
		err = goblas.Zhemv(mat.UploByte(uplo), *n, -one, a, *lda, x.CVector(0, j-1), 1, one, work, 1)

		//        Compute componentwise relative backward error from formula
		//
		//        max(i) ( abs(R(i)) / ( abs(A)*abs(X) + abs(B) )(i) )
		//
		//        where abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
		for i = 1; i <= (*n); i++ {
			rwork.Set(i-1, Cabs1(b.Get(i-1, j-1)))
		}

		//        Compute abs(A)*abs(X) + abs(B).
		if upper {
			for k = 1; k <= (*n); k++ {
				s = zero
				xk = Cabs1(x.Get(k-1, j-1))
				for i = 1; i <= k-1; i++ {
					rwork.Set(i-1, rwork.Get(i-1)+Cabs1(a.Get(i-1, k-1))*xk)
					s = s + Cabs1(a.Get(i-1, k-1))*Cabs1(x.Get(i-1, j-1))
				}
				rwork.Set(k-1, rwork.Get(k-1)+math.Abs(a.GetRe(k-1, k-1))*xk+s)
			}
		} else {
			for k = 1; k <= (*n); k++ {
				s = zero
				xk = Cabs1(x.Get(k-1, j-1))
				rwork.Set(k-1, rwork.Get(k-1)+math.Abs(a.GetRe(k-1, k-1))*xk)
				for i = k + 1; i <= (*n); i++ {
					rwork.Set(i-1, rwork.Get(i-1)+Cabs1(a.Get(i-1, k-1))*xk)
					s = s + Cabs1(a.Get(i-1, k-1))*Cabs1(x.Get(i-1, j-1))
				}
				rwork.Set(k-1, rwork.Get(k-1)+s)
			}
		}
		s = zero
		for i = 1; i <= (*n); i++ {
			if rwork.Get(i-1) > safe2 {
				s = maxf64(s, Cabs1(work.Get(i-1))/rwork.Get(i-1))
			} else {
				s = maxf64(s, (Cabs1(work.Get(i-1))+safe1)/(rwork.Get(i-1)+safe1))
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
			Zhetrs(uplo, n, func() *int { y := 1; return &y }(), af, ldaf, ipiv, work.CMatrix(*n, opts), n, info)
			goblas.Zaxpy(*n, one, work, 1, x.CVector(0, j-1), 1)
			lstres = berr.Get(j - 1)
			count = count + 1
			goto label20
		}

		//        Bound error from formula
		//
		//        norm(X - XTRUE) / norm(X) .le. FERR =
		//        norm( abs(inv(A))*
		//           ( abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) ))) / norm(X)
		//
		//        where
		//          norm(Z) is the magnitude of the largest component of Z
		//          inv(A) is the inverse of A
		//          abs(Z) is the componentwise absolute value of the matrix or
		//             vector Z
		//          NZ is the maximum number of nonzeros in any row of A, plus 1
		//          EPS is machine epsilon
		//
		//        The i-th component of abs(R)+NZ*EPS*(abs(A)*abs(X)+abs(B))
		//        is incremented by SAFE1 if the i-th component of
		//        abs(A)*abs(X) + abs(B) is less than SAFE2.
		//
		//        Use ZLACN2 to estimate the infinity-norm of the matrix
		//           inv(A) * diag(W),
		//        where W = abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) )))
		for i = 1; i <= (*n); i++ {
			if rwork.Get(i-1) > safe2 {
				rwork.Set(i-1, Cabs1(work.Get(i-1))+float64(nz)*eps*rwork.Get(i-1))
			} else {
				rwork.Set(i-1, Cabs1(work.Get(i-1))+float64(nz)*eps*rwork.Get(i-1)+safe1)
			}
		}

		kase = 0
	label100:
		;
		Zlacn2(n, work.Off((*n)+1-1), work, ferr.GetPtr(j-1), &kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Multiply by diag(W)*inv(A**H).
				Zhetrs(uplo, n, func() *int { y := 1; return &y }(), af, ldaf, ipiv, work.CMatrix(*n, opts), n, info)
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, rwork.GetCmplx(i-1)*work.Get(i-1))
				}
			} else if kase == 2 {
				//              Multiply by inv(A)*diag(W).
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, rwork.GetCmplx(i-1)*work.Get(i-1))
				}
				Zhetrs(uplo, n, func() *int { y := 1; return &y }(), af, ldaf, ipiv, work.CMatrix(*n, opts), n, info)
			}
			goto label100
		}

		//        Normalize error.
		lstres = zero
		for i = 1; i <= (*n); i++ {
			lstres = maxf64(lstres, Cabs1(x.Get(i-1, j-1)))
		}
		if lstres != zero {
			ferr.Set(j-1, ferr.Get(j-1)/lstres)
		}

	}
}
