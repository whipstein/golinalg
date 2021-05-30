package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zgbrfs improves the computed solution to a system of linear
// equations when the coefficient matrix is banded, and provides
// error bounds and backward error estimates for the solution.
func Zgbrfs(trans byte, n, kl, ku, nrhs *int, ab *mat.CMatrix, ldab *int, afb *mat.CMatrix, ldafb *int, ipiv *[]int, b *mat.CMatrix, ldb *int, x *mat.CMatrix, ldx *int, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector, info *int) {
	var notran bool
	var transn, transt byte
	var cone complex128
	var eps, lstres, s, safe1, safe2, safmin, three, two, xk, zero float64
	var count, i, itmax, j, k, kase, kk, nz int
	isave := make([]int, 3)

	itmax = 5
	zero = 0.0
	cone = (1.0 + 0.0*1i)
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
	} else if (*kl) < 0 {
		(*info) = -3
	} else if (*ku) < 0 {
		(*info) = -4
	} else if (*nrhs) < 0 {
		(*info) = -5
	} else if (*ldab) < (*kl)+(*ku)+1 {
		(*info) = -7
	} else if (*ldafb) < 2*(*kl)+(*ku)+1 {
		(*info) = -9
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -12
	} else if (*ldx) < maxint(1, *n) {
		(*info) = -14
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGBRFS"), -(*info))
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
	nz = minint((*kl)+(*ku)+2, (*n)+1)
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
		goblas.Zcopy(n, b.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
		goblas.Zgbmv(mat.TransByte(trans), n, n, kl, ku, toPtrc128(-cone), ab, ldab, x.CVector(0, j-1), func() *int { y := 1; return &y }(), &cone, work, func() *int { y := 1; return &y }())

		//        Compute componentwise relative backward error from formula
		//
		//        max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
		//
		//        where abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
		for i = 1; i <= (*n); i++ {
			rwork.Set(i-1, Cabs1(b.Get(i-1, j-1)))
		}

		//        Compute abs(op(A))*abs(X) + abs(B).
		if notran {
			for k = 1; k <= (*n); k++ {
				kk = (*ku) + 1 - k
				xk = Cabs1(x.Get(k-1, j-1))
				for i = maxint(1, k-(*ku)); i <= minint(*n, k+(*kl)); i++ {
					rwork.Set(i-1, rwork.Get(i-1)+Cabs1(ab.Get(kk+i-1, k-1))*xk)
				}
			}
		} else {
			for k = 1; k <= (*n); k++ {
				s = zero
				kk = (*ku) + 1 - k
				for i = maxint(1, k-(*ku)); i <= minint(*n, k+(*kl)); i++ {
					s = s + Cabs1(ab.Get(kk+i-1, k-1))*Cabs1(x.Get(i-1, j-1))
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
			Zgbtrs(trans, n, kl, ku, func() *int { y := 1; return &y }(), afb, ldafb, ipiv, work.CMatrix(*n, opts), n, info)
			goblas.Zaxpy(n, &cone, work, func() *int { y := 1; return &y }(), x.CVector(0, j-1), func() *int { y := 1; return &y }())
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
	label100:
		;
		Zlacn2(n, work.Off((*n)+1-1), work, ferr.GetPtr(j-1), &kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Multiply by diag(W)*inv(op(A)**H).
				Zgbtrs(transt, n, kl, ku, func() *int { y := 1; return &y }(), afb, ldafb, ipiv, work.CMatrix(*n, opts), n, info)
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, complex(rwork.Get(i-1), 0)*work.Get(i-1))
				}
			} else {
				//              Multiply by inv(op(A))*diag(W).
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, complex(rwork.Get(i-1), 0)*work.Get(i-1))
				}
				Zgbtrs(transn, n, kl, ku, func() *int { y := 1; return &y }(), afb, ldafb, ipiv, work.CMatrix(*n, opts), n, info)
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
