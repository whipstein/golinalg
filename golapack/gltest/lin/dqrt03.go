package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dqrt03 tests Dormqr, which computes Q*C, Q'*C, C*Q or C*Q'.
//
// DQRT03 compares the results of a call to Dormqr with the results of
// forming Q explicitly by a call to Dorgqr and then performing matrix
// multiplication by a call to DGEMM.
func dqrt03(m, n, k int, af, c, cc, q *mat.Matrix, tau, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var side, trans byte
	var cnorm, eps, one, resid, rogue float64
	var iside, itrans, j, mc, nc int
	var err error

	iseed := make([]int, 4)
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	rogue = -1.0e+10

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k columns of the factorization to the array Q
	golapack.Dlaset(Full, m, m, rogue, rogue, q)
	golapack.Dlacpy(Lower, m-1, k, af.Off(1, 0), q.Off(1, 0))

	//     Generate the m-by-m matrix Q
	*srnamt = "Dorgqr"
	if err = golapack.Dorgqr(m, m, k, q, tau, work, lwork); err != nil {
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
			golapack.Dlarnv(2, &iseed, mc, c.Off(0, j-1).Vector())
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
			*srnamt = "Dormqr"
			if err = golapack.Dormqr(mat.SideByte(side), mat.TransByte(trans), mc, nc, k, af, tau, cc, work, lwork); err != nil {
				panic(err)
			}

			//           Form explicit product and subtract
			if side == 'L' {
				if err = cc.Gemm(mat.TransByte(trans), mat.NoTrans, mc, nc, mc, -one, q, c, one); err != nil {
					panic(err)
				}
			} else {
				if err = cc.Gemm(mat.NoTrans, mat.TransByte(trans), mc, nc, nc, -one, c, q, one); err != nil {
					panic(err)
				}
			}

			//           Compute error in the difference
			resid = golapack.Dlange('1', mc, nc, cc, rwork)
			result.Set((iside-1)*2+itrans-1, resid/(float64(max(1, m))*cnorm*eps))

		}
	}
}
