package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dqrt15 generates a matrix with full or deficient rank and of various
// norms.
func dqrt15(scale, rksel, m, n, nrhs int, a, b *mat.Matrix, s *mat.Vector, iseed []int, work *mat.Vector, lwork int) (int, float64, float64, []int) {
	var bignum, eps, norma, normb, one, smlnum, svmin, temp, two, zero float64
	var j, mn, rank int
	var err error

	dummy := vf(1)

	zero = 0.0
	one = 1.0
	two = 2.0
	svmin = 0.1

	mn = min(m, n)
	if lwork < max(m+mn, mn*nrhs, 2*n+m) {
		gltest.Xerbla("dqrt15", 16)
		return rank, norma, normb, iseed
	}

	smlnum = golapack.Dlamch(SafeMinimum)
	bignum = one / smlnum
	eps = golapack.Dlamch(Epsilon)
	smlnum = (smlnum / eps) / eps
	bignum = one / smlnum

	//     Determine rank and (unscaled) singular values
	if rksel == 1 {
		rank = mn
	} else if rksel == 2 {
		rank = (3 * mn) / 4
		for j = rank + 1; j <= mn; j++ {
			s.Set(j-1, zero)
		}
	} else {
		gltest.Xerbla("dqrt15", 2)
	}

	if rank > 0 {
		//        Nontrivial case
		s.Set(0, one)
		for j = 2; j <= rank; j++ {
		label20:
			;
			temp = matgen.Dlarnd(1, &iseed)
			if temp > svmin {
				s.Set(j-1, math.Abs(temp))
			} else {
				goto label20
			}
		}
		dlaord('D', rank, s, 1)

		//        Generate 'rank' columns of a random orthogonal matrix in A
		golapack.Dlarnv(2, &iseed, m, work)
		work.Scal(m, one/work.Nrm2(m, 1), 1)
		golapack.Dlaset('F', m, rank, zero, one, a)
		golapack.Dlarf(Left, m, rank, work, 1, two, a, work.Off(m))

		//        workspace used: m+mn
		//
		//        Generate consistent rhs in the range space of A
		golapack.Dlarnv(2, &iseed, rank*nrhs, work)
		err = b.Gemm(NoTrans, NoTrans, m, nrhs, rank, one, a, work.Matrix(rank, opts), zero)

		//        work space used: <= mn *nrhs
		//
		//        generate (unscaled) matrix A
		for j = 1; j <= rank; j++ {
			a.Off(0, j-1).Vector().Scal(m, s.Get(j-1), 1)
		}
		if rank < n {
			golapack.Dlaset(Full, m, n-rank, zero, zero, a.Off(0, rank))
		}
		if err = matgen.Dlaror('R', 'N', m, n, a, &iseed, work); err != nil {
			panic(err)
		}

	} else {
		//        work space used 2*n+m
		//
		//        Generate null matrix and rhs
		for j = 1; j <= mn; j++ {
			s.Set(j-1, zero)
		}
		golapack.Dlaset(Full, m, n, zero, zero, a)
		golapack.Dlaset(Full, m, nrhs, zero, zero, b)

	}

	//     Scale the matrix
	if scale != 1 {
		norma = golapack.Dlange('M', m, n, a, dummy)
		if norma != zero {
			if scale == 2 {
				//              matrix scaled up
				if err = golapack.Dlascl('G', 0, 0, norma, bignum, m, n, a); err != nil {
					panic(err)
				}
				if err = golapack.Dlascl('G', 0, 0, norma, bignum, mn, 1, s.Matrix(mn, opts)); err != nil {
					panic(err)
				}
				if err = golapack.Dlascl('G', 0, 0, norma, bignum, m, nrhs, b); err != nil {
					panic(err)
				}
			} else if scale == 3 {
				//              matrix scaled down
				if err = golapack.Dlascl('G', 0, 0, norma, smlnum, m, n, a); err != nil {
					panic(err)
				}
				if err = golapack.Dlascl('G', 0, 0, norma, smlnum, mn, 1, s.Matrix(mn, opts)); err != nil {
					panic(err)
				}
				if err = golapack.Dlascl('G', 0, 0, norma, smlnum, m, nrhs, b); err != nil {
					panic(err)
				}
			} else {
				gltest.Xerbla("dqrt15", 1)
				return rank, norma, normb, iseed
			}
		}
	}

	norma = s.Asum(mn, 1)
	normb = golapack.Dlange('O', m, nrhs, b, dummy)

	return rank, norma, normb, iseed
}
