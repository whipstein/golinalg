package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// drqt03 tests Dormrq, which computes Q*C, Q'*C, C*Q or C*Q'.
//
// DRQT03 compares the results of a call to Dormrq with the results of
// forming Q explicitly by a call to Dorgrq and then performing matrix
// multiplication by a call to DGEMM.
func drqt03(m, n, k int, af, c, cc, q *mat.Matrix, tau, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var side mat.MatSide
	var trans mat.MatTrans
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

	//     Copy the last k rows of the factorization to the array Q
	golapack.Dlaset(Full, n, n, rogue, rogue, q)
	if k > 0 && n > k {
		golapack.Dlacpy(Full, k, n-k, af.Off(m-k, 0), q.Off(n-k, 0))
	}
	if k > 1 {
		golapack.Dlacpy(Lower, k-1, k-1, af.Off(m-k+2-1, n-k), q.Off(n-k+2-1, n-k))
	}

	//     Generate the n-by-n matrix Q
	*srnamt = "Dorgrq"
	if err = golapack.Dorgrq(n, n, k, q, tau.Off(minmn-k), work, lwork); err != nil {
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
			golapack.Dlarnv(2, &iseed, mc, c.Off(0, j-1).Vector())
		}
		cnorm = golapack.Dlange('1', mc, nc, c, rwork)
		if cnorm == 0.0 {
			cnorm = one
		}

		for itrans = 1; itrans <= 2; itrans++ {
			if itrans == 1 {
				trans = NoTrans
			} else {
				trans = Trans
			}

			//           Copy C
			golapack.Dlacpy(Full, mc, nc, c, cc)

			//           Apply Q or Q' to C
			*srnamt = "Dormrq"
			if k > 0 {
				if err = golapack.Dormrq(side, trans, mc, nc, k, af.Off(m-k, 0), tau.Off(minmn-k), cc, work, lwork); err != nil {
					panic(err)
				}
			}

			//           Form explicit product and subtract
			if side == Left {
				if err = cc.Gemm(trans, NoTrans, mc, nc, mc, -one, q, c, one); err != nil {
					panic(err)
				}
			} else {
				if err = cc.Gemm(NoTrans, trans, mc, nc, nc, -one, c, q, one); err != nil {
					panic(err)
				}
			}

			//           Compute error in the difference
			resid = golapack.Dlange('1', mc, nc, cc, rwork)
			result.Set((iside-1)*2+itrans-1, resid/(float64(max(1, n))*cnorm*eps))

		}
	}
}
