package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zlqt03 tests Zunmlq, which computes Q*C, Q'*C, C*Q or C*Q'.
//
// ZLQT03 compares the results of a call to Zunmlq with the results of
// forming Q explicitly by a call to Zunglq and then performing matrix
// multiplication by a call to ZGEMM.
func zlqt03(m, n, k int, af, c, cc, q *mat.CMatrix, tau, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
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

	//     Copy the first k rows of the factorization to the array Q
	golapack.Zlaset(Full, n, n, rogue, rogue, q)
	golapack.Zlacpy(Upper, k, n-1, af.Off(0, 1), q.Off(0, 1))

	//     Generate the n-by-n matrix Q
	*srnamt = "Zunglq"
	if err = golapack.Zunglq(n, n, k, q, tau, work, lwork); err != nil {
		panic(err)
	}

	for iside = 1; iside <= 2; iside++ {
		if iside == 1 {
			side = Left
			mc = n
			nc = m
		} else {
			side = Right
			mc = m
			nc = n
		}

		//        Generate MC by NC matrix C
		for j = 1; j <= nc; j++ {
			golapack.Zlarnv(2, &iseed, mc, c.CVector(0, j-1))
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
			*srnamt = "Zunmlq"
			if err = golapack.Zunmlq(side, trans, mc, nc, k, af, tau, cc, work, lwork); err != nil {
				panic(err)
			}

			//           Form explicit product and subtract
			if side == Left {
				if err = goblas.Zgemm(trans, NoTrans, mc, nc, mc, complex(-one, 0), q, c, complex(one, 0), cc); err != nil {
					panic(err)
				}
			} else {
				if err = goblas.Zgemm(NoTrans, trans, mc, nc, nc, complex(-one, 0), c, q, complex(one, 0), cc); err != nil {
					panic(err)
				}
			}

			//           Compute error in the difference
			resid = golapack.Zlange('1', mc, nc, cc, rwork)
			result.Set((iside-1)*2+itrans-1, resid/(float64(max(1, n))*cnorm*eps))

		}
	}
}
