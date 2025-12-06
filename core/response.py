from enum import Enum
from typing import Any, Optional
from flask import jsonify


class ResponseCode(str, Enum):
    SUCCESS = '10000'
    FAILURE = '10001'
    RETRY = '10002'
    INVALID_ACCESS_TOKEN = '10003'
    INVALID_REFRESH_TOKEN = '10004'
    FORBIDDEN = '10005'
    UNAUTHORIZED = '10006'
    NOT_FOUND = '10007'
    VALIDATION_ERROR = '10008'
    INTERNAL_ERROR = '10009'
    BAD_REQUEST = '10010'


def create_response(
    success: bool,
    message: str,
    code: ResponseCode,
    data: Optional[Any] = None,
    http_status: int = 200
):
    """
    Tạo response chuẩn theo format yêu cầu.
    
    Args:
        success: True nếu thành công, False nếu thất bại
        message: Thông báo mô tả
        code: Mã ResponseCode
        data: Dữ liệu trả về (optional)
        http_status: HTTP status code (default 200)
    
    Returns:
        Flask response object
    """
    response_body = {
        'success': success,
        'message': message,
        'code': code.value
    }
    
    if data is not None:
        response_body['data'] = data
    
    return jsonify(response_body), http_status


def success_response(message: str = "Thành công", data: Optional[Any] = None):
    """Response thành công"""
    return create_response(
        success=True,
        message=message,
        code=ResponseCode.SUCCESS,
        data=data,
        http_status=200
    )


def error_response(
    message: str,
    code: ResponseCode = ResponseCode.FAILURE,
    http_status: int = 400,
    data: Optional[Any] = None
):
    """Response lỗi"""
    return create_response(
        success=False,
        message=message,
        code=code,
        data=data,
        http_status=http_status
    )


def validation_error(message: str = "Dữ liệu không hợp lệ", data: Optional[Any] = None):
    """Response lỗi validation"""
    return error_response(
        message=message,
        code=ResponseCode.VALIDATION_ERROR,
        http_status=400,
        data=data
    )


def not_found_error(message: str = "Không tìm thấy", data: Optional[Any] = None):
    """Response không tìm thấy"""
    return error_response(
        message=message,
        code=ResponseCode.NOT_FOUND,
        http_status=404,
        data=data
    )


def internal_error(message: str = "Lỗi hệ thống", data: Optional[Any] = None):
    """Response lỗi server"""
    return error_response(
        message=message,
        code=ResponseCode.INTERNAL_ERROR,
        http_status=500,
        data=data
    )


def bad_request(message: str = "Yêu cầu không hợp lệ", data: Optional[Any] = None):
    """Response bad request"""
    return error_response(
        message=message,
        code=ResponseCode.BAD_REQUEST,
        http_status=400,
        data=data
    )
