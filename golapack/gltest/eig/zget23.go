package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zget23 checks the nonsymmetric eigenvalue problem driver CGEEVX.
//    If COMP = .FALSE., the first 8 of the following tests will be
//    performed on the input matrix A, and also test 9 if LWORK is
//    sufficiently large.
//    if COMP is .TRUE. all 11 tests will be performed.
//
//    (1)     | A * VR - VR * W | / ( n |A| ulp )
//
//      Here VR is the matrix of unit right eigenvectors.
//      W is a diagonal matrix with diagonal entries W(j).
//
//    (2)     | A**H * VL - VL * W**H | / ( n |A| ulp )
//
//      Here VL is the matrix of unit left eigenvectors, A**H is the
//      conjugate transpose of A, and W is as above.
//
//    (3)     | |VR(i)| - 1 | / ulp and largest component real
//
//      VR(i) denotes the i-th column of VR.
//
//    (4)     | |VL(i)| - 1 | / ulp and largest component real
//
//      VL(i) denotes the i-th column of VL.
//
//    (5)     0 if W(full) = W(partial), 1/ulp otherwise
//
//      W(full) denotes the eigenvalues computed when VR, VL, RCONDV
//      and RCONDE are also computed, and W(partial) denotes the
//      eigenvalues computed when only some of VR, VL, RCONDV, and
//      RCONDE are computed.
//
//    (6)     0 if VR(full) = VR(partial), 1/ulp otherwise
//
//      VR(full) denotes the right eigenvectors computed when VL, RCONDV
//      and RCONDE are computed, and VR(partial) denotes the result
//      when only some of VL and RCONDV are computed.
//
//    (7)     0 if VL(full) = VL(partial), 1/ulp otherwise
//
//      VL(full) denotes the left eigenvectors computed when VR, RCONDV
//      and RCONDE are computed, and VL(partial) denotes the result
//      when only some of VR and RCONDV are computed.
//
//    (8)     0 if SCALE, ILO, IHI, ABNRM (full) =
//                 SCALE, ILO, IHI, ABNRM (partial)
//            1/ulp otherwise
//
//      SCALE, ILO, IHI and ABNRM describe how the matrix is balanced.
//      (full) is when VR, VL, RCONDE and RCONDV are also computed, and
//      (partial) is when some are not computed.
//
//    (9)     0 if RCONDV(full) = RCONDV(partial), 1/ulp otherwise
//
//      RCONDV(full) denotes the reciprocal condition numbers of the
//      right eigenvectors computed when VR, VL and RCONDE are also
//      computed. RCONDV(partial) denotes the reciprocal condition
//      numbers when only some of VR, VL and RCONDE are computed.
//
//   (10)     |RCONDV - RCDVIN| / cond(RCONDV)
//
//      RCONDV is the reciprocal right eigenvector condition number
//      computed by ZGEEVX and RCDVIN (the precomputed true value)
//      is supplied as input. cond(RCONDV) is the condition number of
//      RCONDV, and takes errors in computing RCONDV into account, so
//      that the resulting quantity should be O(ULP). cond(RCONDV) is
//      essentially given by norm(A)/RCONDE.
//
//   (11)     |RCONDE - RCDEIN| / cond(RCONDE)
//
//      RCONDE is the reciprocal eigenvalue condition number
//      computed by ZGEEVX and RCDEIN (the precomputed true value)
//      is supplied as input.  cond(RCONDE) is the condition number
//      of RCONDE, and takes errors in computing RCONDE into account,
//      so that the resulting quantity should be O(ULP). cond(RCONDE)
//      is essentially given by norm(A)/RCONDV.
func zget23(comp bool, isrt int, balanc byte, jtype int, thresh float64, iseed []int, n int, a, h *mat.CMatrix, w, w1 *mat.CVector, vl, vr, lre *mat.CMatrix, rcondv, rcndv1, rcdvin, rconde, rcnde1, rcdein, scale, scale1, result *mat.Vector, work *mat.CVector, lwork int, rwork *mat.Vector) (err error) {
	var balok, nobal bool
	var sense byte
	var ctmp complex128
	var abnrm, abnrm1, eps, epsin, one, smlnum, tnrm, tol, tolin, two, ulp, ulpinv, v, vmax, vmx, vricmp, vrimin, vrmx, vtst, zero float64
	var i, ihi, ihi1, iinfo, ilo, ilo1, isens, isensm, j, jj, kmin int
	sens := []byte{'N', 'V'}
	cdum := cvf(1)
	res := vf(2)

	zero = 0.0
	one = 1.0
	two = 2.0
	epsin = 5.9605e-8

	//     Check for errors
	nobal = balanc == 'N'
	balok = nobal || balanc == 'P' || balanc == 'S' || balanc == 'B'
	if isrt != 0 && isrt != 1 {
		err = fmt.Errorf("isrt != 0 && isrt != 1: isrt=%v", isrt)
	} else if !balok {
		err = fmt.Errorf("!balok: balanc='%c'", balanc)
	} else if thresh < zero {
		err = fmt.Errorf("thresh < zero: thresh=%v", thresh)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < 1 || a.Rows < n {
		err = fmt.Errorf("a.Rows < 1 || a.Rows < n: a.Rows=%v, n=%v", a.Rows, n)
	} else if vl.Rows < 1 || vl.Rows < n {
		err = fmt.Errorf("vl.Rows < 1 || vl.Rows < n: vl.Rows=%v, n=%v", vl.Rows, n)
	} else if vr.Rows < 1 || vr.Rows < n {
		err = fmt.Errorf("vr.Rows < 1 || vr.Rows < n: vr.Rows=%v, n=%v", vr.Rows, n)
	} else if lre.Rows < 1 || lre.Rows < n {
		err = fmt.Errorf("lre.Rows < 1 || lre.Rows < n: lre.Rows=%v, n=%v", lre.Rows, n)
	} else if lwork < 2*n || (comp && lwork < 2*n+n*n) {
		err = fmt.Errorf("lwork < 2*n || (comp && lwork < 2*n+n*n): lwork=%v, n=%v, comp=%v", lwork, n, comp)
	}

	if err != nil {
		gltest.Xerbla2("zget23", err)
		return
	}

	//     Quick return if nothing to do
	for i = 1; i <= 11; i++ {
		result.Set(i-1, -one)
	}

	if n == 0 {
		return
	}

	//     More Important constants
	ulp = golapack.Dlamch(Precision)
	smlnum = golapack.Dlamch(SafeMinimum)
	ulpinv = one / ulp

	//     Compute eigenvalues and eigenvectors, and test them
	if lwork >= 2*n+n*n {
		sense = 'B'
		isensm = 2
	} else {
		sense = 'E'
		isensm = 1
	}
	golapack.Zlacpy(Full, n, n, a, h)
	if ilo, ihi, abnrm, iinfo, err = golapack.Zgeevx(balanc, 'V', 'V', sense, n, h, w, vl, vr, scale, rconde, rcondv, work, lwork, rwork); err != nil || iinfo != 0 {
		result.Set(0, ulpinv)
		if jtype != 22 {
			fmt.Printf(" zget23: %s returned info=%6d.\n         N=%6d, jtype=%6d, balanc=%c, iseed=%5d\n", "Zgeevx1", iinfo, n, jtype, balanc, iseed)
		} else {
			fmt.Printf(" zget23: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeevx1", iinfo, n, iseed[0])
		}
		err = fmt.Errorf("iinfo=%v", abs(iinfo))
		return
	}

	//     Do Test (1)
	zget22(NoTrans, NoTrans, NoTrans, n, a, vr, w, work, rwork, res)
	result.Set(0, res.Get(0))

	//     Do Test (2)
	zget22(ConjTrans, NoTrans, ConjTrans, n, a, vl, w, work, rwork, res)
	result.Set(1, res.Get(0))

	//     Do Test (3)
	for j = 1; j <= n; j++ {
		tnrm = goblas.Dznrm2(n, vr.CVector(0, j-1, 1))
		result.Set(2, math.Max(result.Get(2), math.Min(ulpinv, math.Abs(tnrm-one)/ulp)))
		vmx = zero
		vrmx = zero
		for jj = 1; jj <= n; jj++ {
			vtst = vr.GetMag(jj-1, j-1)
			if vtst > vmx {
				vmx = vtst
			}
			if vr.GetIm(jj-1, j-1) == zero && math.Abs(vr.GetRe(jj-1, j-1)) > vrmx {
				vrmx = math.Abs(vr.GetRe(jj-1, j-1))
			}
		}
		if vrmx/vmx < one-two*ulp {
			result.Set(2, ulpinv)
		}
	}

	//     Do Test (4)
	for j = 1; j <= n; j++ {
		tnrm = goblas.Dznrm2(n, vl.CVector(0, j-1, 1))
		result.Set(3, math.Max(result.Get(3), math.Min(ulpinv, math.Abs(tnrm-one)/ulp)))
		vmx = zero
		vrmx = zero
		for jj = 1; jj <= n; jj++ {
			vtst = vl.GetMag(jj-1, j-1)
			if vtst > vmx {
				vmx = vtst
			}
			if vl.GetIm(jj-1, j-1) == zero && math.Abs(vl.GetRe(jj-1, j-1)) > vrmx {
				vrmx = math.Abs(vl.GetRe(jj-1, j-1))
			}
		}
		if vrmx/vmx < one-two*ulp {
			result.Set(3, ulpinv)
		}
	}

	//     Test for all options of computing condition numbers
	for isens = 1; isens <= isensm; isens++ {

		sense = sens[isens-1]

		//        Compute eigenvalues only, and test them
		golapack.Zlacpy(Full, n, n, a, h)
		if ilo1, ihi1, abnrm1, iinfo, err = golapack.Zgeevx(balanc, 'N', 'N', sense, n, h, w1, cdum.CMatrix(1, opts), cdum.CMatrix(1, opts), scale1, rcnde1, rcndv1, work, lwork, rwork); err != nil || iinfo != 0 {
			result.Set(0, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget23: %s returned info=%6d.\n         N=%6d, jtype=%6d, balanc=%c, iseed=%5d\n", "Zgeevx2", iinfo, n, jtype, balanc, iseed)
			} else {
				fmt.Printf(" zget23: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeevx2", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label190
		}

		//        Do Test (5)
		for j = 1; j <= n; j++ {
			if w.Get(j-1) != w1.Get(j-1) {
				result.Set(4, ulpinv)
			}
		}

		//        Do Test (8)
		if !nobal {
			for j = 1; j <= n; j++ {
				if scale.Get(j-1) != scale1.Get(j-1) {
					result.Set(7, ulpinv)
				}
			}
			if ilo != ilo1 {
				result.Set(7, ulpinv)
			}
			if ihi != ihi1 {
				result.Set(7, ulpinv)
			}
			if abnrm != abnrm1 {
				result.Set(7, ulpinv)
			}
		}

		//        Do Test (9)
		if isens == 2 && n > 1 {
			for j = 1; j <= n; j++ {
				if rcondv.Get(j-1) != rcndv1.Get(j-1) {
					result.Set(8, ulpinv)
				}
			}
		}

		//        Compute eigenvalues and right eigenvectors, and test them
		golapack.Zlacpy(Full, n, n, a, h)
		if ilo1, ihi1, abnrm1, iinfo, err = golapack.Zgeevx(balanc, 'N', 'V', sense, n, h, w1, cdum.CMatrix(1, opts), lre, scale1, rcnde1, rcndv1, work, lwork, rwork); err != nil || iinfo != 0 {
			result.Set(0, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget23: %s returned info=%6d.\n         N=%6d, jtype=%6d, balanc=%c, iseed=%5d\n", "Zgeevx3", iinfo, n, jtype, balanc, iseed)
			} else {
				fmt.Printf(" zget23: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeevx3", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label190
		}

		//        Do Test (5) again
		for j = 1; j <= n; j++ {
			if w.Get(j-1) != w1.Get(j-1) {
				result.Set(4, ulpinv)
			}
		}

		//        Do Test (6)
		for j = 1; j <= n; j++ {
			for jj = 1; jj <= n; jj++ {
				if vr.Get(j-1, jj-1) != lre.Get(j-1, jj-1) {
					result.Set(5, ulpinv)
				}
			}
		}

		//        Do Test (8) again
		if !nobal {
			for j = 1; j <= n; j++ {
				if scale.Get(j-1) != scale1.Get(j-1) {
					result.Set(7, ulpinv)
				}
			}
			if ilo != ilo1 {
				result.Set(7, ulpinv)
			}
			if ihi != ihi1 {
				result.Set(7, ulpinv)
			}
			if abnrm != abnrm1 {
				result.Set(7, ulpinv)
			}
		}

		//        Do Test (9) again
		if isens == 2 && n > 1 {
			for j = 1; j <= n; j++ {
				if rcondv.Get(j-1) != rcndv1.Get(j-1) {
					result.Set(8, ulpinv)
				}
			}
		}

		//        Compute eigenvalues and left eigenvectors, and test them
		golapack.Zlacpy(Full, n, n, a, h)
		if ilo1, ihi1, abnrm1, iinfo, err = golapack.Zgeevx(balanc, 'V', 'N', sense, n, h, w1, lre, cdum.CMatrix(1, opts), scale1, rcnde1, rcndv1, work, lwork, rwork); err != nil || iinfo != 0 {
			result.Set(0, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget23: %s returned info=%6d.\n         N=%6d, jtype=%6d, balanc=%c, iseed=%5d\n", "Zgeevx4", iinfo, n, jtype, balanc, iseed)
			} else {
				fmt.Printf(" zget23: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeevx4", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label190
		}

		//        Do Test (5) again
		for j = 1; j <= n; j++ {
			if w.Get(j-1) != w1.Get(j-1) {
				result.Set(4, ulpinv)
			}
		}

		//        Do Test (7)
		for j = 1; j <= n; j++ {
			for jj = 1; jj <= n; jj++ {
				if vl.Get(j-1, jj-1) != lre.Get(j-1, jj-1) {
					result.Set(6, ulpinv)
				}
			}
		}

		//        Do Test (8) again
		if !nobal {
			for j = 1; j <= n; j++ {
				if scale.Get(j-1) != scale1.Get(j-1) {
					result.Set(7, ulpinv)
				}
			}
			if ilo != ilo1 {
				result.Set(7, ulpinv)
			}
			if ihi != ihi1 {
				result.Set(7, ulpinv)
			}
			if abnrm != abnrm1 {
				result.Set(7, ulpinv)
			}
		}

		//        Do Test (9) again
		if isens == 2 && n > 1 {
			for j = 1; j <= n; j++ {
				if rcondv.Get(j-1) != rcndv1.Get(j-1) {
					result.Set(8, ulpinv)
				}
			}
		}

	label190:
	}

	//     If COMP, compare condition numbers to precomputed ones
	if comp {
		golapack.Zlacpy(Full, n, n, a, h)
		if ilo, ihi, abnrm, iinfo, err = golapack.Zgeevx('N', 'V', 'V', 'B', n, h, w, vl, vr, scale, rconde, rcondv, work, lwork, rwork); err != nil || iinfo != 0 {
			result.Set(0, ulpinv)
			fmt.Printf(" zget23: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeevx5", iinfo, n, iseed[0])
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label250
		}

		//        Sort eigenvalues and condition numbers lexicographically
		//        to compare with inputs
		for i = 1; i <= n-1; i++ {
			kmin = i
			if isrt == 0 {
				vrimin = w.GetRe(i - 1)
			} else {
				vrimin = w.GetIm(i - 1)
			}
			for j = i + 1; j <= n; j++ {
				if isrt == 0 {
					vricmp = w.GetRe(j - 1)
				} else {
					vricmp = w.GetIm(j - 1)
				}
				if vricmp < vrimin {
					kmin = j
					vrimin = vricmp
				}
			}
			ctmp = w.Get(kmin - 1)
			w.Set(kmin-1, w.Get(i-1))
			w.Set(i-1, ctmp)
			vrimin = rconde.Get(kmin - 1)
			rconde.Set(kmin-1, rconde.Get(i-1))
			rconde.Set(i-1, vrimin)
			vrimin = rcondv.Get(kmin - 1)
			rcondv.Set(kmin-1, rcondv.Get(i-1))
			rcondv.Set(i-1, vrimin)
		}

		//        Compare condition numbers for eigenvectors
		//        taking their condition numbers into account
		result.Set(9, zero)
		eps = math.Max(epsin, ulp)
		v = math.Max(float64(n)*eps*abnrm, smlnum)
		if abnrm == zero {
			v = one
		}
		for i = 1; i <= n; i++ {
			if v > rcondv.Get(i-1)*rconde.Get(i-1) {
				tol = rcondv.Get(i - 1)
			} else {
				tol = v / rconde.Get(i-1)
			}
			if v > rcdvin.Get(i-1)*rcdein.Get(i-1) {
				tolin = rcdvin.Get(i - 1)
			} else {
				tolin = v / rcdein.Get(i-1)
			}
			tol = math.Max(tol, smlnum/eps)
			tolin = math.Max(tolin, smlnum/eps)
			if eps*(rcdvin.Get(i-1)-tolin) > rcondv.Get(i-1)+tol {
				vmax = one / eps
			} else if rcdvin.Get(i-1)-tolin > rcondv.Get(i-1)+tol {
				vmax = (rcdvin.Get(i-1) - tolin) / (rcondv.Get(i-1) + tol)
			} else if rcdvin.Get(i-1)+tolin < eps*(rcondv.Get(i-1)-tol) {
				vmax = one / eps
			} else if rcdvin.Get(i-1)+tolin < rcondv.Get(i-1)-tol {
				vmax = (rcondv.Get(i-1) - tol) / (rcdvin.Get(i-1) + tolin)
			} else {
				vmax = one
			}
			result.Set(9, math.Max(result.Get(9), vmax))
		}

		//        Compare condition numbers for eigenvalues
		//        taking their condition numbers into account
		result.Set(10, zero)
		for i = 1; i <= n; i++ {
			if v > rcondv.Get(i-1) {
				tol = one
			} else {
				tol = v / rcondv.Get(i-1)
			}
			if v > rcdvin.Get(i-1) {
				tolin = one
			} else {
				tolin = v / rcdvin.Get(i-1)
			}
			tol = math.Max(tol, smlnum/eps)
			tolin = math.Max(tolin, smlnum/eps)
			if eps*(rcdein.Get(i-1)-tolin) > rconde.Get(i-1)+tol {
				vmax = one / eps
			} else if rcdein.Get(i-1)-tolin > rconde.Get(i-1)+tol {
				vmax = (rcdein.Get(i-1) - tolin) / (rconde.Get(i-1) + tol)
			} else if rcdein.Get(i-1)+tolin < eps*(rconde.Get(i-1)-tol) {
				vmax = one / eps
			} else if rcdein.Get(i-1)+tolin < rconde.Get(i-1)-tol {
				vmax = (rconde.Get(i-1) - tol) / (rcdein.Get(i-1) + tolin)
			} else {
				vmax = one
			}
			result.Set(10, math.Max(result.Get(10), vmax))
		}
	label250:
	}

	return
}
