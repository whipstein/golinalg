package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dqlt03 tests Dormql, which computes Q*C, Q'*C, C*Q or C*Q'.
//
// DQLT03 compares the results of a call to Dormql with the results of
// forming Q explicitly by a call to DORGQL and then performing matrix
// multiplication by a call to DGEMM.
func dqlt03(m, n, k int, af, c, cc, q *mat.Matrix, tau, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var side, trans byte
	var cnorm, eps, one, resid, rogue, zero float64
	var iside, itrans, j, mc, minmn, nc int
	var err error

	iseed := make([]int, 4)
	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	minmn = min(m, n)

	//     Quick return if possible
	if minmn == 0 {
		result.Set(0, zero)
		result.Set(1, zero)
		result.Set(2, zero)
		result.Set(3, zero)
		return
	}

	//     Copy the last k columns of the factorization to the array Q
	golapack.Dlaset(Full, m, m, rogue, rogue, q)
	if k > 0 && m > k {
		golapack.Dlacpy(Full, m-k, k, af.Off(0, n-k), q.Off(0, m-k))
	}
	if k > 1 {
		golapack.Dlacpy(Upper, k-1, k-1, af.Off(m-k, n-k+2-1), q.Off(m-k, m-k+2-1))
	}

	//     Generate the m-by-m matrix Q
	*srnamt = "Dorgql"
	if err = golapack.Dorgql(m, m, k, q, tau.Off(minmn-k), work, lwork); err != nil {
		panic(err)
	}

	for iside = 1; iside <= 2; iside++ {
		if iside == 1 {
			side = 'L'
			mc = m
			nc = n
		} else {
			side = 'R'
			mc = n
			nc = m
		}

		//        Generate MC by NC matrix C
		for j = 1; j <= nc; j++ {
			golapack.Dlarnv(2, &iseed, mc, c.Vector(0, j-1))
		}
		cnorm = golapack.Dlange('1', mc, nc, c, rwork)
		if cnorm == 0.0 {
			cnorm = one
		}

		for itrans = 1; itrans <= 2; itrans++ {
			if itrans == 1 {
				trans = 'N'
			} else {
				trans = 'T'
			}

			//           Copy C
			golapack.Dlacpy(Full, mc, nc, c, cc)

			//           Apply Q or Q' to C
			*srnamt = "Dormql"
			if k > 0 {
				if err = golapack.Dormql(mat.SideByte(side), mat.TransByte(trans), mc, nc, k, af.Off(0, n-k), tau.Off(minmn-k), cc, work, lwork); err != nil {
					panic(err)
				}
			}

			//           Form explicit product and subtract
			if side == 'L' {
				if err = goblas.Dgemm(mat.TransByte(trans), mat.NoTrans, mc, nc, mc, -one, q, c, one, cc); err != nil {
					panic(err)
				}
			} else {
				if err = goblas.Dgemm(mat.NoTrans, mat.TransByte(trans), mc, nc, nc, -one, c, q, one, cc); err != nil {
					panic(err)
				}
			}

			//           Compute error in the difference
			resid = golapack.Dlange('1', mc, nc, cc, rwork)
			result.Set((iside-1)*2+itrans-1, resid/(float64(max(1, m))*cnorm*eps))

		}
	}
}
