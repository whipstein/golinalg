package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zqrt03 tests Zunmqr, which computes Q*C, Q'*C, C*Q or C*Q'.
//
// ZQRT03 compares the results of a call to Zunmqr with the results of
// forming Q explicitly by a call to Zungqr and then performing matrix
// multiplication by a call to ZGEMM.
func zqrt03(m, n, k int, af, c, cc, q *mat.CMatrix, tau, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
	var side mat.MatSide
	var trans mat.MatTrans
	var rogue complex128
	var cnorm, eps, one, resid, zero float64
	var iside, itrans, j, mc, nc int
	var err error

	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)

	srnamt := &gltest.Common.Srnamc.Srnamt

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k columns of the factorization to the array Q
	golapack.Zlaset(Full, m, m, rogue, rogue, q)
	golapack.Zlacpy(Lower, m-1, k, af.Off(1, 0), q.Off(1, 0))

	//     Generate the m-by-m matrix Q
	*srnamt = "Zungqr"
	if err = golapack.Zungqr(m, m, k, q, tau, work, lwork); err != nil {
		panic(err)
	}

	for iside = 1; iside <= 2; iside++ {
		if iside == 1 {
			side = Left
			mc = m
			nc = n
		} else {
			side = Right
			mc = n
			nc = m
		}

		//        Generate MC by NC matrix C
		for j = 1; j <= nc; j++ {
			golapack.Zlarnv(2, &iseed, mc, c.Off(0, j-1).CVector())
		}
		cnorm = golapack.Zlange('1', mc, nc, c, rwork)
		if cnorm == zero {
			cnorm = one
		}

		for itrans = 1; itrans <= 2; itrans++ {
			if itrans == 1 {
				trans = NoTrans
			} else {
				trans = ConjTrans
			}

			//           Copy C
			golapack.Zlacpy(Full, mc, nc, c, cc)

			//           Apply Q or Q' to C
			*srnamt = "Zunmqr"
			if err = golapack.Zunmqr(side, trans, mc, nc, k, af, tau, cc, work, lwork); err != nil {
				panic(err)
			}

			//           Form explicit product and subtract
			if side == Left {
				if err = cc.Gemm(trans, NoTrans, mc, nc, mc, complex(-one, 0), q, c, complex(one, 0)); err != nil {
					panic(err)
				}
			} else {
				if err = cc.Gemm(NoTrans, trans, mc, nc, nc, complex(-one, 0), c, q, complex(one, 0)); err != nil {
					panic(err)
				}
			}

			//           Compute error in the difference
			resid = golapack.Zlange('1', mc, nc, cc, rwork)
			result.Set((iside-1)*2+itrans-1, resid/(float64(max(1, m))*cnorm*eps))

		}
	}
}
