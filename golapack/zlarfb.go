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
					goblas.Zcopy(n, c.CVector(j-1, 0), ldc, work.CVector(0, j-1), func() *int { y := 1; return &y }())
					Zlacgv(n, work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V1
				goblas.Ztrmm(Right, Lower, NoTrans, Unit, n, k, &one, v, ldv, work, ldwork)
				if (*m) > (*k) {
					//                 W := W + C2**H * V2
					goblas.Zgemm(ConjTrans, NoTrans, n, k, toPtr((*m)-(*k)), &one, c.Off((*k)+1-1, 0), ldc, v.Off((*k)+1-1, 0), ldv, &one, work, ldwork)
				}

				//              W := W * T**H  or  W * T
				goblas.Ztrmm(Right, Upper, mat.TransByte(transt), NonUnit, n, k, &one, t, ldt, work, ldwork)

				//              C := C - V * W**H
				if (*m) > (*k) {
					//                 C2 := C2 - V2 * W**H
					goblas.Zgemm(NoTrans, ConjTrans, toPtr((*m)-(*k)), n, k, toPtrc128(-one), v.Off((*k)+1-1, 0), ldv, work, ldwork, &one, c.Off((*k)+1-1, 0), ldc)
				}

				//              W := W * V1**H
				goblas.Ztrmm(Right, Lower, ConjTrans, Unit, n, k, &one, v, ldv, work, ldwork)
				//
				//              C1 := C1 - W**H
				//
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
					goblas.Zcopy(m, c.CVector(0, j-1), func() *int { y := 1; return &y }(), work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V1
				goblas.Ztrmm(Right, Lower, NoTrans, Unit, m, k, &one, v, ldv, work, ldwork)
				if (*n) > (*k) {
					//                 W := W + C2 * V2
					goblas.Zgemm(NoTrans, NoTrans, m, k, toPtr((*n)-(*k)), &one, c.Off(0, (*k)+1-1), ldc, v.Off((*k)+1-1, 0), ldv, &one, work, ldwork)
				}

				//              W := W * T  or  W * T**H
				goblas.Ztrmm(Right, Upper, mat.TransByte(trans), NonUnit, m, k, &one, t, ldt, work, ldwork)

				//              C := C - W * V**H
				if (*n) > (*k) {
					//                 C2 := C2 - W * V2**H
					goblas.Zgemm(NoTrans, ConjTrans, m, toPtr((*n)-(*k)), k, toPtrc128(-one), work, ldwork, v.Off((*k)+1-1, 0), ldv, &one, c.Off(0, (*k)+1-1), ldc)
				}

				//              W := W * V1**H
				goblas.Ztrmm(Right, Lower, ConjTrans, Unit, m, k, &one, v, ldv, work, ldwork)

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
					goblas.Zcopy(n, c.CVector((*m)-(*k)+j-1, 0), ldc, work.CVector(0, j-1), func() *int { y := 1; return &y }())
					Zlacgv(n, work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V2
				goblas.Ztrmm(Right, Upper, NoTrans, Unit, n, k, &one, v.Off((*m)-(*k)+1-1, 0), ldv, work, ldwork)
				if (*m) > (*k) {
					//                 W := W + C1**H * V1
					goblas.Zgemm(ConjTrans, NoTrans, n, k, toPtr((*m)-(*k)), &one, c, ldc, v, ldv, &one, work, ldwork)
				}

				//              W := W * T**H  or  W * T
				goblas.Ztrmm(Right, Lower, mat.TransByte(transt), NonUnit, n, k, &one, t, ldt, work, ldwork)

				//              C := C - V * W**H
				if (*m) > (*k) {
					//                 C1 := C1 - V1 * W**H
					goblas.Zgemm(NoTrans, ConjTrans, toPtr((*m)-(*k)), n, k, toPtrc128(-one), v, ldv, work, ldwork, &one, c, ldc)
				}

				//              W := W * V2**H
				goblas.Ztrmm(Right, Upper, ConjTrans, Unit, n, k, &one, v.Off((*m)-(*k)+1-1, 0), ldv, work, ldwork)

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
					goblas.Zcopy(m, c.CVector(0, (*n)-(*k)+j-1), func() *int { y := 1; return &y }(), work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V2
				goblas.Ztrmm(Right, Upper, NoTrans, Unit, m, k, &one, v.Off((*n)-(*k)+1-1, 0), ldv, work, ldwork)
				if (*n) > (*k) {
					//                 W := W + C1 * V1
					goblas.Zgemm(NoTrans, NoTrans, m, k, toPtr((*n)-(*k)), &one, c, ldc, v, ldv, &one, work, ldwork)
				}

				//              W := W * T  or  W * T**H
				goblas.Ztrmm(Right, Lower, mat.TransByte(trans), NonUnit, m, k, &one, t, ldt, work, ldwork)

				//              C := C - W * V**H
				if (*n) > (*k) {
					//                 C1 := C1 - W * V1**H
					goblas.Zgemm(NoTrans, ConjTrans, m, toPtr((*n)-(*k)), k, toPtrc128(-one), work, ldwork, v, ldv, &one, c, ldc)
				}

				//              W := W * V2**H
				goblas.Ztrmm(Right, Upper, ConjTrans, Unit, m, k, &one, v.Off((*n)-(*k)+1-1, 0), ldv, work, ldwork)

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
					goblas.Zcopy(n, c.CVector(j-1, 0), ldc, work.CVector(0, j-1), func() *int { y := 1; return &y }())
					Zlacgv(n, work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V1**H
				goblas.Ztrmm(Right, Upper, ConjTrans, Unit, n, k, &one, v, ldv, work, ldwork)
				if (*m) > (*k) {
					//                 W := W + C2**H * V2**H
					goblas.Zgemm(ConjTrans, ConjTrans, n, k, toPtr((*m)-(*k)), &one, c.Off((*k)+1-1, 0), ldc, v.Off(0, (*k)+1-1), ldv, &one, work, ldwork)
				}

				//              W := W * T**H  or  W * T
				goblas.Ztrmm(Right, Upper, mat.TransByte(transt), NonUnit, n, k, &one, t, ldt, work, ldwork)

				//              C := C - V**H * W**H
				if (*m) > (*k) {
					//                 C2 := C2 - V2**H * W**H
					goblas.Zgemm(ConjTrans, ConjTrans, toPtr((*m)-(*k)), n, k, toPtrc128(-one), v.Off(0, (*k)+1-1), ldv, work, ldwork, &one, c.Off((*k)+1-1, 0), ldc)
				}

				//              W := W * V1
				goblas.Ztrmm(Right, Upper, NoTrans, Unit, n, k, &one, v, ldv, work, ldwork)

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
					goblas.Zcopy(m, c.CVector(0, j-1), func() *int { y := 1; return &y }(), work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V1**H
				goblas.Ztrmm(Right, Upper, ConjTrans, Unit, m, k, &one, v, ldv, work, ldwork)
				if (*n) > (*k) {
					//                 W := W + C2 * V2**H
					goblas.Zgemm(NoTrans, ConjTrans, m, k, toPtr((*n)-(*k)), &one, c.Off(0, (*k)+1-1), ldc, v.Off(0, (*k)+1-1), ldv, &one, work, ldwork)
				}

				//              W := W * T  or  W * T**H
				goblas.Ztrmm(Right, Upper, mat.TransByte(trans), NonUnit, m, k, &one, t, ldt, work, ldwork)

				//              C := C - W * V
				if (*n) > (*k) {
					//                 C2 := C2 - W * V2
					goblas.Zgemm(NoTrans, NoTrans, m, toPtr((*n)-(*k)), k, toPtrc128(-one), work, ldwork, v.Off(0, (*k)+1-1), ldv, &one, c.Off(0, (*k)+1-1), ldc)
				}

				//              W := W * V1
				goblas.Ztrmm(Right, Upper, NoTrans, Unit, m, k, &one, v, ldv, work, ldwork)

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
					goblas.Zcopy(n, c.CVector((*m)-(*k)+j-1, 0), ldc, work.CVector(0, j-1), func() *int { y := 1; return &y }())
					Zlacgv(n, work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V2**H
				goblas.Ztrmm(Right, Lower, ConjTrans, Unit, n, k, &one, v.Off(0, (*m)-(*k)+1-1), ldv, work, ldwork)
				if (*m) > (*k) {
					//                 W := W + C1**H * V1**H
					goblas.Zgemm(ConjTrans, ConjTrans, n, k, toPtr((*m)-(*k)), &one, c, ldc, v, ldv, &one, work, ldwork)
				}

				//              W := W * T**H  or  W * T
				goblas.Ztrmm(Right, Lower, mat.TransByte(transt), NonUnit, n, k, &one, t, ldt, work, ldwork)

				//              C := C - V**H * W**H
				if (*m) > (*k) {
					//                 C1 := C1 - V1**H * W**H
					goblas.Zgemm(ConjTrans, ConjTrans, toPtr((*m)-(*k)), n, k, toPtrc128(-one), v, ldv, work, ldwork, &one, c, ldc)
				}

				//              W := W * V2
				goblas.Ztrmm(Right, Lower, NoTrans, Unit, n, k, &one, v.Off(0, (*m)-(*k)+1-1), ldv, work, ldwork)

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
					goblas.Zcopy(m, c.CVector(0, (*n)-(*k)+j-1), func() *int { y := 1; return &y }(), work.CVector(0, j-1), func() *int { y := 1; return &y }())
				}

				//              W := W * V2**H
				goblas.Ztrmm(Right, Lower, ConjTrans, Unit, m, k, &one, v.Off(0, (*n)-(*k)+1-1), ldv, work, ldwork)
				if (*n) > (*k) {
					//                 W := W + C1 * V1**H
					goblas.Zgemm(NoTrans, ConjTrans, m, k, toPtr((*n)-(*k)), &one, c, ldc, v, ldv, &one, work, ldwork)
				}

				//              W := W * T  or  W * T**H
				goblas.Ztrmm(Right, Lower, mat.TransByte(trans), NonUnit, m, k, &one, t, ldt, work, ldwork)

				//              C := C - W * V
				if (*n) > (*k) {
					//                 C1 := C1 - W * V1
					goblas.Zgemm(NoTrans, NoTrans, m, toPtr((*n)-(*k)), k, toPtrc128(-one), work, ldwork, v, ldv, &one, c, ldc)
				}

				//              W := W * V2
				goblas.Ztrmm(Right, Lower, NoTrans, Unit, m, k, &one, v.Off(0, (*n)-(*k)+1-1), ldv, work, ldwork)

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
