package eig

import (
	"github.com/whipstein/golinalg/golapack"
)

// zget36 tests ZTREXC, a routine for reordering diagonal entries of a
// matrix in complex Schur form. Thus, ZLAEXC computes a unitary matrix
// Q such that
//
//    Q' * T1 * Q  = T2
//
// and where one of the diagonal blocks of T1 (the one at row IFST) has
// been moved to position ILST.
//
// The test code verifies that the residual Q'*T1*Q-T2 is small, that T2
// is in Schur form, and that the final position of the IFST block is
// ILST.
//
// The test matrices are read from a file with logical unit number NIN.
func zget36() (rmax float64, lmax, ninfo, knt int) {
	var cone, ctemp, czero complex128
	var eps, one, res, zero float64
	var _i, i, ifst, ilst, info1, info2, j, ldt, lwork, n int
	var err error

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	ldt = 10
	lwork = 2 * ldt * ldt
	diag := cvf(10)
	work := cvf(lwork)
	result := vf(2)
	rwork := vf(10)
	q := cmf(10, 10, opts)
	t1 := cmf(10, 10, opts)
	t2 := cmf(10, 10, opts)
	tmp := cmf(10, 10, opts)

	eps = golapack.Dlamch(Precision)

	nlist := []int{1, 3, 4, 4, 4, 5, 4, 6}
	ifstlist := []int{1, 1, 4, 4, 1, 5, 4, 5}
	ilstlist := []int{1, 3, 1, 1, 4, 1, 1, 3}
	tmplist := [][]complex128{
		{
			0.0e0 + 0.0e0i,
		},
		{
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 2.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 3.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 4.0e0 + 0.0e0i,
		},
		{
			12.0e0 + 0.0e0i, 0.0e0 + 20.0e0i, -2.0e0 + 0.0e0i, 10.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 20.0e0 + 0.0e0i, 2.0e0 + -1.0e0i, 0.0e0 + 0.9e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 3.0e0 + 0.0e0i, 0.8e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 8.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 1.0e0i, 2.0e0 + -1.0e0i, 2.0e0 + -3.0e0i, 12.0e0 + 3.0e0i, 2.0e0 + 39.0e0i,
			0.0e0 + 0.0e0i, 2.0e0 + 3.0e0i, 2.0e0 + 3.0e0i, 2.0e0 + 13.0e0i, 2.0e0 + 31.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, -2.0e0 + 3.0e0i, 2.0e0 + 3.0e0i, 12.0e0 + 3.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 2.0e0 + -3.0e0i, -2.0e0 + 3.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 2.0e0 + 3.0e0i,
		},
		{
			0.0621e0 + 0.7054e0i, 0.1062e0 + 0.0503e0i, 0.6553e0 + 0.5876e0i, 0.2560e0 + 0.8642e0i,
			0.0e0 + 0.0e0i, 0.2640e0 + 0.5782e0i, 0.9700e0 + 0.7256e0i, 0.5598e0 + 0.1943e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0380e0 + 0.2849e0i, 0.9166e0 + 0.0580e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.1402e0 + 0.6908e0i,
		},
		{
			10.0e0 + 1.0e0i, 10.0e0 + 0.0e0i, 30.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 10.0e0 + 1.0e0i, 10.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 20.0e0 + 1.0e0i, 30.0e0 + 0.0e0i, 20.0e0 + 1.0e0i, 0.0e0 + -1.0e0i, 0.0e0 + -10.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 30.0e0 + 1.0e0i, 0.0e0 + 0.0e0i, 2.0e0 + 0.0e0i, 0.0e0 + 20.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 40.0e0 + 1.0e0i, 0.0e0 + -10.0e0i, -30.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 50.0e0 + 1.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 60.0e0 + 1.0e0i,
		},
	}
	for _i, n = range nlist {
		ifst = ifstlist[_i]
		ilst = ilstlist[_i]

		knt = knt + 1
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				tmp.Set(i-1, j-1, tmplist[_i][(i-1)*(n)+j-1])
			}
		}
		golapack.Zlacpy(Full, n, n, tmp, t1)
		golapack.Zlacpy(Full, n, n, tmp, t2)
		res = zero

		//     Test without accumulating Q
		golapack.Zlaset(Full, n, n, czero, cone, q)
		if err = golapack.Ztrexc('N', n, t1, q, ifst, ilst); err != nil {
			panic(err)
		}
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				if i == j && q.Get(i-1, j-1) != cone {
					res = res + one/eps
				}
				if i != j && q.Get(i-1, j-1) != czero {
					res = res + one/eps
				}
			}
		}

		//     Test with accumulating Q
		golapack.Zlaset(Full, n, n, czero, cone, q)
		if err = golapack.Ztrexc('V', n, t2, q, ifst, ilst); err != nil {
			panic(err)
		}

		//     Compare T1 with T2
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				if t1.Get(i-1, j-1) != t2.Get(i-1, j-1) {
					res = res + one/eps
				}
			}
		}
		if info1 != 0 || info2 != 0 {
			ninfo = ninfo + 1
		}
		if info1 != info2 {
			res = res + one/eps
		}

		//     Test for successful reordering of T2
		diag.Copy(n, tmp.Off(0, 0).CVector(), ldt+1, 1)
		if ifst < ilst {
			for i = ifst + 1; i <= ilst; i++ {
				ctemp = diag.Get(i - 1)
				diag.Set(i-1, diag.Get(i-1-1))
				diag.Set(i-1-1, ctemp)
			}
		} else if ifst > ilst {
			for i = ifst - 1; i >= ilst; i-- {
				ctemp = diag.Get(i + 1 - 1)
				diag.Set(i, diag.Get(i-1))
				diag.Set(i-1, ctemp)
			}
		}
		for i = 1; i <= n; i++ {
			if t2.Get(i-1, i-1) != diag.Get(i-1) {
				res = res + one/eps
			}
		}

		//     Test for small residual, and orthogonality of Q
		zhst01(n, 1, n, tmp, t2, q, work, lwork, rwork, result)
		res = res + result.Get(0) + result.Get(1)

		//     Test for T2 being in Schur form
		for j = 1; j <= n-1; j++ {
			for i = j + 1; i <= n; i++ {
				if t2.Get(i-1, j-1) != czero {
					res = res + one/eps
				}
			}
		}
		if res > rmax {
			rmax = res
			lmax = knt
		}
	}

	return
}
