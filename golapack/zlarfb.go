package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlarfb applies a complex block reflector H or its transpose H**H to a
// complex M-by-N matrix C, from either the left or the right.
func Zlarfb(side, trans, direct, storev byte, m, n, k *int, v *mat.CMatrix, ldv *int, t *mat.CMatrix, ldt *int, c *mat.CMatrix, ldc *int, work *mat.CMatrix, ldwork *int) {
	var transt byte
	var one complex128
	var i, j int
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	if trans == 'N' {
		transt = 'C'
	} else {
		transt = 'N'
	}

	if storev == 'C' {

		if direct == 'F' {
			//           Let  V =  ( V1 )    (first K rows)
			//                     ( V2 )
			//           where  V1  is unit lower triangular.
			if side == 'L' {
				//              Form  H * C  or  H**H * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**H * V  =  (C1**H * V1 + C2**H * V2)  (stored in WORK)
				//
				//              W := C1**H
				for j = 1; j <= (*k); j++ {
					goblas.Zcopy(*n, c.CVector(j-1, 0, *ldc), work.CVector(0, j-1, 1))
					Zlacgv(n, work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V1
				err = goblas.Ztrmm(Right, Lower, NoTrans, Unit, *n, *k, one, v, work)
				if (*m) > (*k) {
					//                 W := W + C2**H * V2
					err = goblas.Zgemm(ConjTrans, NoTrans, *n, *k, (*m)-(*k), one, c.Off((*k), 0), v.Off((*k), 0), one, work)
				}

				//              W := W * T**H  or  W * T
				err = goblas.Ztrmm(Right, Upper, mat.TransByte(transt), NonUnit, *n, *k, one, t, work)

				//              C := C - V * W**H
				if (*m) > (*k) {
					//                 C2 := C2 - V2 * W**H
					err = goblas.Zgemm(NoTrans, ConjTrans, (*m)-(*k), *n, *k, -one, v.Off((*k), 0), work, one, c.Off((*k), 0))
				}

				//              W := W * V1**H
				err = goblas.Ztrmm(Right, Lower, ConjTrans, Unit, *n, *k, one, v, work)

				//              C1 := C1 - W**H
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*n); i++ {
						c.Set(j-1, i-1, c.Get(j-1, i-1)-work.GetConj(i-1, j-1))
					}
				}

			} else if side == 'R' {
				//              Form  C * H  or  C * H**H  where  C = ( C1  C2 )
				//
				//              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
				//
				//              W := C1
				for j = 1; j <= (*k); j++ {
					goblas.Zcopy(*m, c.CVector(0, j-1, 1), work.CVector(0, j-1, 1))
				}

				//              W := W * V1
				err = goblas.Ztrmm(Right, Lower, NoTrans, Unit, *m, *k, one, v, work)
				if (*n) > (*k) {
					//                 W := W + C2 * V2
					err = goblas.Zgemm(NoTrans, NoTrans, *m, *k, (*n)-(*k), one, c.Off(0, (*k)), v.Off((*k), 0), one, work)
				}

				//              W := W * T  or  W * T**H
				err = goblas.Ztrmm(Right, Upper, mat.TransByte(trans), NonUnit, *m, *k, one, t, work)

				//              C := C - W * V**H
				if (*n) > (*k) {
					//                 C2 := C2 - W * V2**H
					err = goblas.Zgemm(NoTrans, ConjTrans, *m, (*n)-(*k), *k, -one, work, v.Off((*k), 0), one, c.Off(0, (*k)))
				}

				//              W := W * V1**H
				err = goblas.Ztrmm(Right, Lower, ConjTrans, Unit, *m, *k, one, v, work)

				//              C1 := C1 - W
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*m); i++ {
						c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(i-1, j-1))
					}
				}
			}

		} else {
			//           Let  V =  ( V1 )
			//                     ( V2 )    (last K rows)
			//           where  V2  is unit upper triangular.
			if side == 'L' {
				//              Form  H * C  or  H**H * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**H * V  =  (C1**H * V1 + C2**H * V2)  (stored in WORK)
				//
				//              W := C2**H
				for j = 1; j <= (*k); j++ {
					goblas.Zcopy(*n, c.CVector((*m)-(*k)+j-1, 0, *ldc), work.CVector(0, j-1, 1))
					Zlacgv(n, work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V2
				err = goblas.Ztrmm(Right, Upper, NoTrans, Unit, *n, *k, one, v.Off((*m)-(*k), 0), work)
				if (*m) > (*k) {
					//                 W := W + C1**H * V1
					err = goblas.Zgemm(ConjTrans, NoTrans, *n, *k, (*m)-(*k), one, c, v, one, work)
				}

				//              W := W * T**H  or  W * T
				err = goblas.Ztrmm(Right, Lower, mat.TransByte(transt), NonUnit, *n, *k, one, t, work)

				//              C := C - V * W**H
				if (*m) > (*k) {
					//                 C1 := C1 - V1 * W**H
					err = goblas.Zgemm(NoTrans, ConjTrans, (*m)-(*k), *n, *k, -one, v, work, one, c)
				}

				//              W := W * V2**H
				err = goblas.Ztrmm(Right, Upper, ConjTrans, Unit, *n, *k, one, v.Off((*m)-(*k), 0), work)

				//              C2 := C2 - W**H
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*n); i++ {
						c.Set((*m)-(*k)+j-1, i-1, c.Get((*m)-(*k)+j-1, i-1)-work.GetConj(i-1, j-1))
					}
				}

			} else if side == 'R' {
				//              Form  C * H  or  C * H**H  where  C = ( C1  C2 )
				//
				//              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
				//
				//              W := C2
				for j = 1; j <= (*k); j++ {
					goblas.Zcopy(*m, c.CVector(0, (*n)-(*k)+j-1, 1), work.CVector(0, j-1, 1))
				}

				//              W := W * V2
				err = goblas.Ztrmm(Right, Upper, NoTrans, Unit, *m, *k, one, v.Off((*n)-(*k), 0), work)
				if (*n) > (*k) {
					//                 W := W + C1 * V1
					err = goblas.Zgemm(NoTrans, NoTrans, *m, *k, (*n)-(*k), one, c, v, one, work)
				}

				//              W := W * T  or  W * T**H
				err = goblas.Ztrmm(Right, Lower, mat.TransByte(trans), NonUnit, *m, *k, one, t, work)

				//              C := C - W * V**H
				if (*n) > (*k) {
					//                 C1 := C1 - W * V1**H
					err = goblas.Zgemm(NoTrans, ConjTrans, *m, (*n)-(*k), *k, -one, work, v, one, c)
				}

				//              W := W * V2**H
				err = goblas.Ztrmm(Right, Upper, ConjTrans, Unit, *m, *k, one, v.Off((*n)-(*k), 0), work)

				//              C2 := C2 - W
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*m); i++ {
						c.Set(i-1, (*n)-(*k)+j-1, c.Get(i-1, (*n)-(*k)+j-1)-work.Get(i-1, j-1))
					}
				}
			}
		}

	} else if storev == 'R' {

		if direct == 'F' {
			//           Let  V =  ( V1  V2 )    (V1: first K columns)
			//           where  V1  is unit upper triangular.
			if side == 'L' {
				//              Form  H * C  or  H**H * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**H * V**H  =  (C1**H * V1**H + C2**H * V2**H) (stored in WORK)
				//
				//              W := C1**H
				for j = 1; j <= (*k); j++ {
					goblas.Zcopy(*n, c.CVector(j-1, 0, *ldc), work.CVector(0, j-1, 1))
					Zlacgv(n, work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V1**H
				err = goblas.Ztrmm(Right, Upper, ConjTrans, Unit, *n, *k, one, v, work)
				if (*m) > (*k) {
					//                 W := W + C2**H * V2**H
					err = goblas.Zgemm(ConjTrans, ConjTrans, *n, *k, (*m)-(*k), one, c.Off((*k), 0), v.Off(0, (*k)), one, work)
				}

				//              W := W * T**H  or  W * T
				err = goblas.Ztrmm(Right, Upper, mat.TransByte(transt), NonUnit, *n, *k, one, t, work)

				//              C := C - V**H * W**H
				if (*m) > (*k) {
					//                 C2 := C2 - V2**H * W**H
					err = goblas.Zgemm(ConjTrans, ConjTrans, (*m)-(*k), *n, *k, -one, v.Off(0, (*k)), work, one, c.Off((*k), 0))
				}

				//              W := W * V1
				err = goblas.Ztrmm(Right, Upper, NoTrans, Unit, *n, *k, one, v, work)

				//              C1 := C1 - W**H
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*n); i++ {
						c.Set(j-1, i-1, c.Get(j-1, i-1)-work.GetConj(i-1, j-1))
					}
				}

			} else if side == 'R' {
				//              Form  C * H  or  C * H**H  where  C = ( C1  C2 )
				//
				//              W := C * V**H  =  (C1*V1**H + C2*V2**H)  (stored in WORK)
				//
				//              W := C1
				for j = 1; j <= (*k); j++ {
					goblas.Zcopy(*m, c.CVector(0, j-1, 1), work.CVector(0, j-1, 1))
				}

				//              W := W * V1**H
				err = goblas.Ztrmm(Right, Upper, ConjTrans, Unit, *m, *k, one, v, work)
				if (*n) > (*k) {
					//                 W := W + C2 * V2**H
					err = goblas.Zgemm(NoTrans, ConjTrans, *m, *k, (*n)-(*k), one, c.Off(0, (*k)), v.Off(0, (*k)), one, work)
				}

				//              W := W * T  or  W * T**H
				err = goblas.Ztrmm(Right, Upper, mat.TransByte(trans), NonUnit, *m, *k, one, t, work)

				//              C := C - W * V
				if (*n) > (*k) {
					//                 C2 := C2 - W * V2
					err = goblas.Zgemm(NoTrans, NoTrans, *m, (*n)-(*k), *k, -one, work, v.Off(0, (*k)), one, c.Off(0, (*k)))
				}

				//              W := W * V1
				err = goblas.Ztrmm(Right, Upper, NoTrans, Unit, *m, *k, one, v, work)

				//              C1 := C1 - W
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*m); i++ {
						c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(i-1, j-1))
					}
				}

			}

		} else {
			//           Let  V =  ( V1  V2 )    (V2: last K columns)
			//           where  V2  is unit lower triangular.
			if side == 'L' {
				//              Form  H * C  or  H**H * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**H * V**H  =  (C1**H * V1**H + C2**H * V2**H) (stored in WORK)
				//
				//              W := C2**H
				for j = 1; j <= (*k); j++ {
					goblas.Zcopy(*n, c.CVector((*m)-(*k)+j-1, 0, *ldc), work.CVector(0, j-1, 1))
					Zlacgv(n, work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V2**H
				err = goblas.Ztrmm(Right, Lower, ConjTrans, Unit, *n, *k, one, v.Off(0, (*m)-(*k)), work)
				if (*m) > (*k) {
					//                 W := W + C1**H * V1**H
					err = goblas.Zgemm(ConjTrans, ConjTrans, *n, *k, (*m)-(*k), one, c, v, one, work)
				}

				//              W := W * T**H  or  W * T
				err = goblas.Ztrmm(Right, Lower, mat.TransByte(transt), NonUnit, *n, *k, one, t, work)

				//              C := C - V**H * W**H
				if (*m) > (*k) {
					//                 C1 := C1 - V1**H * W**H
					err = goblas.Zgemm(ConjTrans, ConjTrans, (*m)-(*k), *n, *k, -one, v, work, one, c)
				}

				//              W := W * V2
				err = goblas.Ztrmm(Right, Lower, NoTrans, Unit, *n, *k, one, v.Off(0, (*m)-(*k)), work)

				//              C2 := C2 - W**H
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*n); i++ {
						c.Set((*m)-(*k)+j-1, i-1, c.Get((*m)-(*k)+j-1, i-1)-work.GetConj(i-1, j-1))
					}
				}

			} else if side == 'R' {
				//              Form  C * H  or  C * H**H  where  C = ( C1  C2 )
				//
				//              W := C * V**H  =  (C1*V1**H + C2*V2**H)  (stored in WORK)
				//
				//              W := C2
				for j = 1; j <= (*k); j++ {
					goblas.Zcopy(*m, c.CVector(0, (*n)-(*k)+j-1, 1), work.CVector(0, j-1, 1))
				}

				//              W := W * V2**H
				err = goblas.Ztrmm(Right, Lower, ConjTrans, Unit, *m, *k, one, v.Off(0, (*n)-(*k)), work)
				if (*n) > (*k) {
					//                 W := W + C1 * V1**H
					err = goblas.Zgemm(NoTrans, ConjTrans, *m, *k, (*n)-(*k), one, c, v, one, work)
				}

				//              W := W * T  or  W * T**H
				err = goblas.Ztrmm(Right, Lower, mat.TransByte(trans), NonUnit, *m, *k, one, t, work)

				//              C := C - W * V
				if (*n) > (*k) {
					//                 C1 := C1 - W * V1
					err = goblas.Zgemm(NoTrans, NoTrans, *m, (*n)-(*k), *k, -one, work, v, one, c)
				}

				//              W := W * V2
				err = goblas.Ztrmm(Right, Lower, NoTrans, Unit, *m, *k, one, v.Off(0, (*n)-(*k)), work)

				//              C1 := C1 - W
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*m); i++ {
						c.Set(i-1, (*n)-(*k)+j-1, c.Get(i-1, (*n)-(*k)+j-1)-work.Get(i-1, j-1))
					}
				}

			}

		}
	}
}
