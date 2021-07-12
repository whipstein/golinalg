package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dptrfs improves the computed solution to a system of linear
// equations when the coefficient matrix is symmetric positive definite
// and tridiagonal, and provides error bounds and backward error
// estimates for the solution.
func Dptrfs(n, nrhs *int, d, e, df, ef *mat.Vector, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, ferr, berr, work *mat.Vector, info *int) {
	var bi, cx, dx, eps, ex, lstres, one, s, safe1, safe2, safmin, three, two, zero float64
	var count, i, itmax, ix, j, nz int

	itmax = 5
	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0

	//     Test the input parameters.
	(*info) = 0
	if (*n) < 0 {
		(*info) = -1
	} else if (*nrhs) < 0 {
		(*info) = -2
	} else if (*ldb) < max(1, *n) {
		(*info) = -8
	} else if (*ldx) < max(1, *n) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPTRFS"), -(*info))
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
		//        Compute residual R = B - A * X.  Also compute
		//        math.Abs(A)*math.Abs(x) + math.Abs(b) for use in the backward error bound.
		if (*n) == 1 {
			bi = b.Get(0, j-1)
			dx = d.Get(0) * x.Get(0, j-1)
			work.Set((*n), bi-dx)
			work.Set(0, math.Abs(bi)+math.Abs(dx))
		} else {
			bi = b.Get(0, j-1)
			dx = d.Get(0) * x.Get(0, j-1)
			ex = e.Get(0) * x.Get(1, j-1)
			work.Set((*n), bi-dx-ex)
			work.Set(0, math.Abs(bi)+math.Abs(dx)+math.Abs(ex))
			for i = 2; i <= (*n)-1; i++ {
				bi = b.Get(i-1, j-1)
				cx = e.Get(i-1-1) * x.Get(i-1-1, j-1)
				dx = d.Get(i-1) * x.Get(i-1, j-1)
				ex = e.Get(i-1) * x.Get(i, j-1)
				work.Set((*n)+i-1, bi-cx-dx-ex)
				work.Set(i-1, math.Abs(bi)+math.Abs(cx)+math.Abs(dx)+math.Abs(ex))
			}
			bi = b.Get((*n)-1, j-1)
			cx = e.Get((*n)-1-1) * x.Get((*n)-1-1, j-1)
			dx = d.Get((*n)-1) * x.Get((*n)-1, j-1)
			work.Set((*n)+(*n)-1, bi-cx-dx)
			work.Set((*n)-1, math.Abs(bi)+math.Abs(cx)+math.Abs(dx))
		}

		//        Compute componentwise relative backward error from formula
		//
		//        max(i) ( math.Abs(R(i)) / ( math.Abs(A)*math.Abs(X) + math.Abs(B) )(i) )
		//
		//        where math.Abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
		s = zero
		for i = 1; i <= (*n); i++ {
			if work.Get(i-1) > safe2 {
				s = math.Max(s, math.Abs(work.Get((*n)+i-1))/work.Get(i-1))
			} else {
				s = math.Max(s, (math.Abs(work.Get((*n)+i-1))+safe1)/(work.Get(i-1)+safe1))
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
			Dpttrs(n, func() *int { y := 1; return &y }(), df, ef, work.MatrixOff((*n), *n, opts), n, info)
			goblas.Daxpy(*n, one, work.Off((*n), 1), x.Vector(0, j-1, 1))
			lstres = berr.Get(j - 1)
			count = count + 1
			goto label20
		}

		//        Bound error from formula
		//
		//        norm(X - XTRUE) / norm(X) .le. FERR =
		//        norm( math.Abs(inv(A))*
		//           ( math.Abs(R) + NZ*EPS*( math.Abs(A)*math.Abs(X)+math.Abs(B) ))) / norm(X)
		//
		//        where
		//          norm(Z) is the magnitude of the largest component of Z
		//          inv(A) is the inverse of A
		//          math.Abs(Z) is the componentwise absolute value of the matrix or
		//             vector Z
		//          NZ is the maximum number of nonzeros in any row of A, plus 1
		//          EPS is machine epsilon
		//
		//        The i-th component of math.Abs(R)+NZ*EPS*(math.Abs(A)*math.Abs(X)+math.Abs(B))
		//        is incremented by SAFE1 if the i-th component of
		//        math.Abs(A)*math.Abs(X) + math.Abs(B) is less than SAFE2.
		for i = 1; i <= (*n); i++ {
			if work.Get(i-1) > safe2 {
				work.Set(i-1, math.Abs(work.Get((*n)+i-1))+float64(nz)*eps*work.Get(i-1))
			} else {
				work.Set(i-1, math.Abs(work.Get((*n)+i-1))+float64(nz)*eps*work.Get(i-1)+safe1)
			}
		}
		ix = goblas.Idamax(*n, work.Off(0, 1))
		ferr.Set(j-1, work.Get(ix-1))

		//        Estimate the norm of inv(A).
		//
		//        Solve M(A) * x = e, where M(A) = (m(i,j)) is given by
		//
		//           m(i,j) =  math.Abs(A(i,j)), i = j,
		//           m(i,j) = -math.Abs(A(i,j)), i .ne. j,
		//
		//        and e = [ 1, 1, ..., 1 ]**T.  Note M(A) = M(L)*D*M(L)**T.
		//
		//        Solve M(L) * x = e.
		work.Set(0, one)
		for i = 2; i <= (*n); i++ {
			work.Set(i-1, one+work.Get(i-1-1)*math.Abs(ef.Get(i-1-1)))
		}

		//        Solve D * M(L)**T * x = b.
		work.Set((*n)-1, work.Get((*n)-1)/df.Get((*n)-1))
		for i = (*n) - 1; i >= 1; i-- {
			work.Set(i-1, work.Get(i-1)/df.Get(i-1)+work.Get(i)*math.Abs(ef.Get(i-1)))
		}

		//        Compute norm(inv(A)) = max(x(i)), 1<=i<=n.
		ix = goblas.Idamax(*n, work.Off(0, 1))
		ferr.Set(j-1, ferr.Get(j-1)*math.Abs(work.Get(ix-1)))

		//        Normalize error.
		lstres = zero
		for i = 1; i <= (*n); i++ {
			lstres = math.Max(lstres, math.Abs(x.Get(i-1, j-1)))
		}
		if lstres != zero {
			ferr.Set(j-1, ferr.Get(j-1)/lstres)
		}

	}
}
